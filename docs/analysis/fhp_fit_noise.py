"""Offline study of the FHP advantage fit (analysis script, 2026-10-08; results in docs/paper-fidelity.md).

Refits seat 0's regret net on the advantage memory of a trainer checkpoint under different recipes, two seeds each, and
reports in chips (big blind = 100):
  noise   - how far two refits on the same data disagree (rms difference / sqrt 2), at the 676 pre-flop infosets
            (hand class x betting node; t-weighted group means) and on 200k held-out flop rows;
  vs tab  - rms distance of a net's pre-flop group means from the tabular t-weighted mean of the samples themselves
            (what the fit estimates there; its own standard error is printed);
  vs ref  - rms distance from the mean of the 16,000-step fits;
  held-out wMSE - the training objective on 5 % held-out rows, relative to the first recipe.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python docs/analysis/fhp_fit_noise.py runs/<fhp run>/checkpoint.pt [name filter,...]

Needs a GPU with ~3 GB free and a checkpoint of ``headsup.deepcfr.train --game fhp`` (four-bet game, obs[:79] features).
"""
import sys, time, copy, math, numpy as np, torch
from headsup.model import BaseModel

def legal79(x):
    """Legal (fold, call, raise) of seat 0's FHP decision from the 79 stored features (the raise counter is the 80th):
    pre-flop it always faces a bet and may raise until four actions were made; on the flop it acts second."""
    pre = x[:, 21] == 0
    k0, k1 = x[:, 32:43:2].sum(1), x[:, 44:55:2].sum(1)
    fold = pre | ~((k1 == 1) & (x[:, 43] == 0))
    rais = torch.where(pre, k0 < 4, k1 < 5)
    return torch.stack([fold, torch.ones_like(fold), rais], 1).float()
from headsup.deepcfr.train import loss_weight_scale

dev = "cuda:0"
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
which = sys.argv[2].split(",") if len(sys.argv) > 2 else None
cfg = dict(ck["model_config"]); T = int(ck["iteration"])
m = ck["adv_memory"][0]; n = int(m["size"])
g0 = torch.Generator().manual_seed(0)
perm = torch.randperm(n, generator=g0)
held, train = perm[: n // 20], perm[n // 20:]
def take(idx):
    return (torch.cat([m["obs_int"][idx].float(), m["obs_float"][idx]], 1).to(dev), m["t"][idx].to(dev), m["target"][idx].to(dev))
X, Tt, Y = take(train)
Xh, Th, Yh = take(held)
ntr = len(X)
scale = float(Y[:65536].pow(2).mean().sqrt())
print(f"iteration {T}: {n:,} samples of seat 0, target rms {scale:.1f}", flush=True)

# ---- pre-flop groups (hand class x betting node) on the training rows: the tabular estimate of what the net fits
pre = (X[:, 21] == 0).nonzero().flatten()
Xp, Tp, Yp = X[pre], Tt[pre], Y[pre]
c1, c2 = Xp[:, 2].long() - 1, Xp[:, 5].long() - 1
r1, r2, s1, s2 = c1 % 13, c2 % 13, c1 // 13, c2 // 13
cls = torch.maximum(r1, r2) * 26 + torch.minimum(r1, r2) * 2 + (s1 == s2).long()
hist = (Xp[:, 31:43] * 1000).round().long()
node = (hist * torch.arange(1, 13, device=dev) * 7919).sum(1)
key = cls * 1_000_003 + node % 1_000_003
uk, inv = torch.unique(key, return_inverse=True)
G = len(uk)
first = torch.zeros(G, dtype=torch.long, device=dev).scatter_reduce(0, inv, torch.arange(len(inv), device=dev), "amin", include_self=False)
legal_g = legal79(Xp[first])
W = torch.zeros(G, device=dev).index_add_(0, inv, Tp)
def gmean(v):  # t-weighted group means of per-row values (rows, A)
    return torch.zeros(G, v.shape[1], device=dev).index_add_(0, inv, Tp[:, None] * v) / W[:, None]
tab = gmean(Yp)
se2 = torch.zeros(G, 3, device=dev).index_add_(0, inv, (Tp[:, None] * (Yp - tab[inv])) ** 2) / W[:, None] ** 2
wg = (W / W.sum())[:, None] * legal_g
wg = wg / wg.sum()
rmsg = lambda d: float((wg * d ** 2).sum().sqrt())
print(f"pre-flop: {len(pre):,} rows in {G} infosets (class x node); tabular mean regret rms {rmsg(tab):.1f}, its standard error {float((wg * se2).sum().sqrt()):.1f}", flush=True)
# ---- flop rows of the held-out part
fl = (Xh[:, 21] == 1).nonzero().flatten()[:200_000]
Xf, Tf = Xh[fl], Th[fl]
legal_f = legal79(Xf)
wf = Tf[:, None] * legal_f; wf = wf / wf.sum()
rmsf = lambda d: float((wf * d ** 2).sum().sqrt())

@torch.no_grad()
def predict(model, x):
    return torch.cat([model(x[i:i + 65536]) for i in range(0, len(x), 65536)])

def evaluate(model):
    return dict(pre=gmean(predict(model, Xp)), flop=predict(model, Xf),
                held=float((Th[:, None] * (predict(model, Xh) - Yh) ** 2).sum() / Th.sum() / 3))

def fit(seed, steps=4000, batch=10000, lr=1e-3, weights="paper", zero_head=False, ema=0.0, sched=None, config=cfg, clip=1.0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model = BaseModel(config=config, zero_head=zero_head).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ws = loss_weight_scale(T, 1.0, weights)
    avg, norms = None, []
    for step in range(steps):
        if sched == "cosine":
            for gq in opt.param_groups: gq["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        elif sched == "drop" and step == int(0.75 * steps):
            for gq in opt.param_groups: gq["lr"] = lr * 0.1
        idx = torch.randint(0, ntr, (batch,), device=dev, generator=gen)
        loss = (ws * Tt[idx][:, None] * (model(X[idx]) - Y[idx] / scale).pow(2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else 1e9)
        if step % 50 == 0: norms.append(float(gn))
        opt.step()
        if ema and step >= steps // 2:
            if avg is None:
                avg = copy.deepcopy(model)
            else:
                with torch.no_grad():
                    for pa, p in zip(avg.parameters(), model.parameters()): pa.mul_(ema).add_(p, alpha=1 - ema)
    out = []
    for mdl in ([model] + ([avg] if avg is not None else [])):
        mdl.eval()
        with torch.no_grad():
            mdl.action_head.weight.mul_(scale); mdl.action_head.bias.mul_(scale)
        out.append(evaluate(mdl))
    return out, float(np.median(norms)), float(np.mean(np.asarray(norms) > (clip or 1e9)))

old_arch = dict(cfg, arch="paper")
VARIANTS = {
    "paper weights, random head (current)": dict(ema=0.998),
    "raw weights (clipped every step), random head": dict(weights="raw"),
    "paper weights, zero head": dict(zero_head=True),
    "raw weights, zero head (before the audit)": dict(weights="raw", zero_head=True),
    "old network (--net paper), raw, zero head": dict(weights="raw", zero_head=True, config=old_arch),
    "cosine learning rate": dict(sched="cosine"),
    "learning rate / 10 for the last quarter": dict(sched="drop"),
    "batch 40,000": dict(batch=40000, ema=0.998),
    "batch 40,000, cosine": dict(batch=40000, sched="cosine"),
    "batch 20,000, cosine": dict(batch=20000, sched="cosine"),
    "8,000 steps, cosine": dict(steps=8000, sched="cosine"),
    "batch 40,000, cosine from 3e-3": dict(batch=40000, sched="cosine", lr=3e-3),
    "cosine from 3e-3": dict(sched="cosine", lr=3e-3),
    "16,000 steps": dict(steps=16000, ema=0.998),
    "16,000 steps, cosine": dict(steps=16000, sched="cosine"),
}
res = {}
for name, kw in VARIANTS.items():
    if which and not any(w in name for w in which): continue
    t0 = time.time()
    runs = [fit(seed, **kw) for seed in (0, 1)]
    res[name] = [r[0][0] for r in runs]
    if kw.get("ema"):
        res[name + " + weight average (EMA 0.998, 2nd half)"] = [r[0][1] for r in runs]
    print(f"fitted {name}: {time.time() - t0:.0f}s, median gradient norm {runs[0][1]:.2f}, clipped steps {runs[0][2]:.0%}", flush=True)
long = [e for k, v in res.items() if k.startswith("16,000") for e in v]
ref_pre = torch.stack([e["pre"] for e in long]).mean(0) if long else None
ref_flop = torch.stack([e["flop"] for e in long]).mean(0) if long else None
base = res[next(iter(res))][0]["held"]
print(f"\n{'recipe':66s} {'noise pre':>9s} {'vs tab':>7s} {'vs ref':>7s} | {'noise flop':>10s} {'vs ref':>7s} | held-out wMSE")
for name, (a, b) in res.items():
    npre = rmsg(a["pre"] - b["pre"]) / math.sqrt(2)
    etab = np.mean([rmsg(e["pre"] - tab) for e in (a, b)])
    nfl = rmsf(a["flop"] - b["flop"]) / math.sqrt(2)
    rp = np.mean([rmsg(e["pre"] - ref_pre) for e in (a, b)]) if long else float("nan")
    rf = np.mean([rmsf(e["flop"] - ref_flop) for e in (a, b)]) if long else float("nan")
    print(f"{name:66s} {npre:9.1f} {etab:7.1f} {rp:7.1f} | {nfl:10.1f} {rf:7.1f} | {np.mean([a['held'], b['held']]) / base:.4f}")
print("noise = rms difference of two refits / sqrt(2); 'vs tab' = rms distance to the tabular mean (its own standard error is above);")
print("'vs ref' = rms distance to the mean of the four 16,000-step fits (they contain two of the reference's own members).")
