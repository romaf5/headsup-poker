"""How much the paper's fit (4,000 steps of 10,000) gains from more samples: seat 0's advantage memory of an FHP checkpoint,
training subsets of growing size, one fixed held-out set.  Chips; the held-out loss is relative to the largest subset's.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python docs/analysis/fhp_fit_datasize.py <checkpoint.pt> <steps> <rows>[,<rows>...]

Results of 2026-10-10: docs/paper-fidelity.md.
"""
import math, sys, time
import torch
from headsup.deepcfr.train import loss_weight_scale
from headsup.model import BaseModel

dev = "cuda:0"
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
steps = int(sys.argv[2]); sizes = [int(float(x)) for x in sys.argv[3].split(",")]
cfg = dict(ck["model_config"]); T = int(ck["iteration"])
m = ck["adv_memory"][0]; n_all = int(m["size"])
perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))
held_i, pool_i = perm[:400_000].sort().values, perm[400_000:400_000 + max(sizes)]
take = lambda idx: (torch.cat([m["obs_int"][idx].float(), m["obs_float"][idx]], 1).to(dev), m["t"][idx].to(dev), m["target"][idx].to(dev))  # noqa: E731
Xh, Th, Yh = take(held_i)
X, Tt, Y = take(pool_i)   # in random order: the first k rows are a random subset
scale = float(Y[:65536].pow(2).mean().sqrt()); ws = loss_weight_scale(T, 1.0, "paper")
flop = (Xh[:, 21] == 1)
print(f"iteration {T}: pool {len(X):,} of {n_all:,} rows, held-out {len(Xh):,} ({int(flop.sum()):,} on the flop); target rms {scale:.1f}; {steps} steps", flush=True)

def fit(k, seed):
    torch.manual_seed(seed); gen = torch.Generator(device=dev).manual_seed(seed)
    model = BaseModel(config=cfg, zero_head=False).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, k, (10000,), device=dev, generator=gen)
        loss = (ws * Tt[idx][:, None] * (model(X[idx]) - Y[idx] / scale).pow(2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(Xh[i:i + 65536]) for i in range(0, len(Xh), 65536)]) * scale
        tr = torch.cat([model(X[i:i + 65536]) for i in range(0, min(k, 400_000), 65536)]) * scale
    w = Th[:, None]
    return p, float((w * (p - Yh) ** 2).sum() / w.sum() / 3), float((Tt[:len(tr), None] * (tr - Y[:len(tr)]) ** 2).sum() / Tt[:len(tr)].sum() / 3)

res = {}
for k in sorted(sizes, reverse=True):
    t0 = time.time()
    (pa, ha, ta), (pb, hb, tb) = fit(k, 0), fit(k, 1)
    wf = Th[flop][:, None]; d = (pa - pb)[flop]
    res[k] = ((ha + hb) / 2, (ta + tb) / 2, float(((wf * d ** 2).sum() / wf.sum() / 3).sqrt()) / math.sqrt(2),
              float(((Th[~flop][:, None] * (pa - pb)[~flop] ** 2).sum() / Th[~flop].sum() / 3).sqrt()) / math.sqrt(2))
    base = res[max(sizes)][0]
    print(f"{k:>11,} rows ({steps * 10000 / k:5.1f} passes): held-out loss {res[k][0] / base:.4f}  train loss {res[k][1] / base:.4f}  "
          f"seed noise flop {res[k][2]:5.1f}  pre-flop {res[k][3]:5.1f} chips   ({time.time() - t0:.0f}s)", flush=True)
