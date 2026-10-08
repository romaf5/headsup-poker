"""Does a suit-canonical input help the FHP advantage fit?  Same memory (seat 0 of a checkpoint), the paper's recipe
(4,000 steps of 10,000, Adam 1e-3, clip 1), raw observations against observations whose cards are mapped to the
lexicographically smallest of their 24 suit-permuted images."""
import sys, itertools, math, time, numpy as np, torch
from headsup.model import BaseModel
from headsup.deepcfr.train import loss_weight_scale

dev = "cuda:0"
PERMS = torch.tensor(list(itertools.permutations(range(4))), device=dev)  # (24, 4)

def canonical(x):
    """obs (N, >=21) -> the same rows with hole cards / flop / turn / river re-suited to the smallest image (ids sorted
    inside the hole pair and the flop, as the engine stores them); empty slots stay empty."""
    x = x.clone()
    ids = x[:, 2:21:3].long() - 1                      # (N, 7) card ids, -1 = no card
    have = ids >= 0
    rank, suit = ids.clamp(min=0) % 13, ids.clamp(min=0) // 13
    best_key, best = None, None
    for g in PERMS:
        new = torch.where(have, rank + 13 * g[suit], torch.full_like(ids, 99))
        hole, _ = new[:, :2].sort(dim=1)
        flop, _ = new[:, 2:5].sort(dim=1)
        cand = torch.cat([hole, flop, new[:, 5:]], dim=1)
        key = (cand * (100 ** torch.arange(6, -1, -1, device=dev))).sum(1)
        if best is None:
            best_key, best = key, cand
        else:
            take = key < best_key
            best_key = torch.where(take, key, best_key)
            best = torch.where(take[:, None], cand, best)
    ok = best < 99
    b = best.clamp(max=51)
    x[:, 0:21:3] = torch.where(ok, b % 13 + 1, torch.zeros_like(b)).float()
    x[:, 1:21:3] = torch.where(ok, b // 13 + 1, torch.zeros_like(b)).float()
    x[:, 2:21:3] = torch.where(ok, b + 1, torch.zeros_like(b)).float()
    return x

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
cfg = dict(ck["model_config"]); T = int(ck["iteration"])
m = ck["adv_memory"][0]; n = int(m["size"])
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
held, train = perm[: n // 20], perm[n // 20:]
def take(idx):
    return (torch.cat([m["obs_int"][idx].float(), m["obs_float"][idx]], 1).to(dev), m["t"][idx].to(dev), m["target"][idx].to(dev))
X, Tt, Y = take(train); Xh, Th, Yh = take(held)
# sanity: canonicalisation is idempotent and keeps the multiset of ranks
Xc = torch.cat([canonical(X[i:i + 500000]) for i in range(0, len(X), 500000)]); Xhc = canonical(Xh)
assert torch.equal(canonical(Xc[:100000]), Xc[:100000])
assert torch.equal(Xc[:100000, 0:6:3].sort(1)[0], X[:100000, 0:6:3].sort(1)[0])
print(f"iteration {T}: {n:,} samples; rows changed by the canonical map: {float((Xc != X).any(1).float().mean()):.1%}; "
      f"distinct card inputs raw {len(torch.unique(X[:, 2:21:3], dim=0)):,} -> canonical {len(torch.unique(Xc[:, 2:21:3], dim=0)):,}", flush=True)
scale = float(Y[:65536].pow(2).mean().sqrt())
pre_h, flop_h = Xh[:, 21] == 0, Xh[:, 21] == 1

def fit(seed, x, steps=4000, batch=10000, lr=1e-3):
    torch.manual_seed(seed); gen = torch.Generator(device=dev).manual_seed(seed)
    model = BaseModel(config=cfg, zero_head=False).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr); ws = loss_weight_scale(T, 1.0, "paper")
    for step in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=dev, generator=gen)
        loss = (ws * Tt[idx][:, None] * (model(x[idx]) - Y[idx] / scale).pow(2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        model.action_head.weight.mul_(scale); model.action_head.bias.mul_(scale)
    return model

@torch.no_grad()
def predict(model, x):
    return torch.cat([model(x[i:i + 65536]) for i in range(0, len(x), 65536)])

def wmse(pred, sel):
    return float((Th[sel, None] * (pred[sel] - Yh[sel]) ** 2).sum() / Th[sel].sum() / 3)

res = {}
for name, xtr, xho in (("raw cards (as trained)", X, Xh), ("suit-canonical cards", Xc, Xhc)):
    t0 = time.time()
    preds = [predict(fit(seed, xtr), xho) for seed in (0, 1)]
    res[name] = preds
    noise = lambda sel: float(((Th[sel, None] * (preds[0][sel] - preds[1][sel]) ** 2).sum() / Th[sel].sum() / 3).sqrt()) / math.sqrt(2)
    print(f"{name:26s} held-out wMSE all {np.mean([wmse(p, slice(None)) for p in preds]):9.1f}  pre-flop {np.mean([wmse(p, pre_h) for p in preds]):9.1f}  "
          f"flop {np.mean([wmse(p, flop_h) for p in preds]):9.1f} | refit noise (chips rms) pre-flop {noise(pre_h):5.1f}  flop {noise(flop_h):5.1f}  ({time.time() - t0:.0f}s)", flush=True)
