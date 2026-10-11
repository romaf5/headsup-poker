"""Network variants of the advantage fit on the same rows (paired): seat 0's memory of an FHP checkpoint, the paper's fit
(4,000 steps of 10,000 by default), one fixed held-out set.  Variants: wider networks, the authors' skip connection around
the middle card layer, rank + suit embeddings without the 52-card table.  Reports the held-out loss relative to the first
variant with the paired standard error, the seed-to-seed noise on held-out flop rows (chips) and the ReLU units that are
active on fewer than 1 % of rows.  Results of 2026-10-10: docs/paper-fidelity.md.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python docs/analysis/fhp_fit_capacity.py <checkpoint.pt> <rows> <steps> [variant filter,...]
"""
import math, sys, time
import torch
from headsup.deepcfr.train import loss_weight_scale
from headsup.model import BaseModel, CardModel

dev = "cuda:0"
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
rows, steps = int(float(sys.argv[2])), int(sys.argv[3])
which = sys.argv[4].split(",") if len(sys.argv) > 4 else None
cfg = dict(ck["model_config"]); T = int(ck["iteration"])
m = ck["adv_memory"][0]; n_all = int(m["size"])
perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))
held_i, pool_i = perm[:400_000].sort().values, perm[400_000:400_000 + rows]
take = lambda idx: (torch.cat([m["obs_int"][idx].float(), m["obs_float"][idx]], 1).to(dev), m["t"][idx].to(dev), m["target"][idx].to(dev))  # noqa: E731
Xh, Th, Yh = take(held_i); X, Tt, Y = take(pool_i)
scale = float(Y[:65536].pow(2).mean().sqrt()); ws = loss_weight_scale(T, 1.0, "paper")
flop = (Xh[:, 21] == 1)
print(f"iteration {T}: {len(X):,} training rows, held-out {len(Xh):,}; target rms {scale:.1f}; {steps} steps", flush=True)

def skip_forward(self, cards):  # the authors' card branch: a skip connection around the middle layer
    groups = []
    for emb, slots in zip(self.group_embeddings, self.groups):
        total = emb(cards[:, slots[0]])
        for s in slots[1:]:
            total = total + emb(cards[:, s])
        groups.append(total)
    x = self.act(self.fc1(torch.cat(groups, dim=1)))
    x = self.act(self.fc2(x) + x)
    return self.act(self.fc3(x))

def make(dim=64, card_skip=False, emb_std=None, no_card=False):
    model = BaseModel(config=dict(cfg, dim=dim), zero_head=False)
    if no_card:  # rank + suit embeddings only: what the paper's parameter count (23 d^2 + 74 d + 4) contains
        for e in model.card_model.group_embeddings:
            torch.nn.init.zeros_(e.card_embedding.weight)
            e.card_embedding.weight.requires_grad_(False)
    if card_skip:
        model.card_model.forward = skip_forward.__get__(model.card_model, CardModel)
    if emb_std is not None:
        for e in model.card_model.group_embeddings:
            for t in (e.rank_embedding, e.suit_embedding, e.card_embedding):
                torch.nn.init.normal_(t.weight, std=emb_std)
    return model

def fit(seed, **kw):
    torch.manual_seed(seed); gen = torch.Generator(device=dev).manual_seed(seed)
    model = make(**kw).to(dev); model.train()
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (10000,), device=dev, generator=gen)
        loss = (ws * Tt[idx][:, None] * (model(X[idx]) - Y[idx] / scale).pow(2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    acts = {}
    hooks = [getattr(model.card_model, l).register_forward_hook(lambda mod, i, o, l=l: acts.__setitem__(l, (o > 0).float().mean(0))) for l in ("fc2", "fc3")]
    hooks += [getattr(model, l).register_forward_hook(lambda mod, i, o, l=l: acts.__setitem__(l, (o > 0).float().mean(0))) for l in ("comb1", "comb2")]
    with torch.no_grad():
        p = torch.cat([model(Xh[i:i + 65536]) for i in range(0, len(Xh), 65536)]) * scale
        model(Xh[:65536])
    for h in hooks: h.remove()
    dead = {k: f"{int((v < 0.01).sum())}/{len(v)}" for k, v in acts.items()}
    se = (Th[:, None] * (p - Yh) ** 2).sum(1) / 3
    return p, se, dead, sum(q.numel() for q in model.parameters())

VARIANTS = {"dim 64 (--net deepcfr)": {}, "dim 64, rank + suit embeddings only (no 52-card table)": dict(no_card=True),
            "dim 64, no 52-card table, card skip": dict(no_card=True, card_skip=True), "dim 64, skip around the middle card layer (authors' code)": dict(card_skip=True),
            "dim 128": dict(dim=128), "dim 128, card skip": dict(dim=128, card_skip=True), "dim 256": dict(dim=256),
            "dim 64, embeddings initialised with std 0.1": dict(emb_std=0.1)}
base = None
for name, kw in VARIANTS.items():
    if which and not any(w in name for w in which): continue
    t0 = time.time()
    (pa, sa, dead, npar), (pb, sb, _, _) = fit(0, **kw), fit(1, **kw)
    s = (sa + sb) / 2
    if base is None: base = s
    rel = float(s.sum() / base.sum())
    d = (s - base) / (base.sum() / len(base))  # paired per-row differences, in units of the baseline's mean loss
    err = float(d.std() / math.sqrt(len(d)))
    wf = Th[flop][:, None]; dd = (pa - pb)[flop]
    noise = float(((wf * dd ** 2).sum() / wf.sum() / 3).sqrt()) / math.sqrt(2)
    print(f"{name:58s} {npar:9,d} params  held-out {rel:.4f} (+- {err:.4f} paired)  flop seed noise {noise:5.1f}  never-active units {dead}   ({time.time() - t0:.0f}s)", flush=True)
