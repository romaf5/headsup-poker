"""Our advantage network against the authors' (DREAM repository: MainPokerModuleFLAT + DuelingQNet), fitted to the same
samples: seat 0's advantage memory of one of our FHP checkpoints, the paper's fit (4,000 steps of 10,000, Adam 1e-3, clip 1,
weights 2t/T), two seeds each.  Reported in chips, on the legal actions:
  noise    - rms disagreement of the two seeds / sqrt 2 at the pre-flop infosets (hand class x betting node) and on held-out flop rows
  vs tab   - rms distance of the pre-flop group means from the tabular t-weighted mean of the samples (its standard error is printed)
  held-out - the weighted squared error on 5 % held-out rows, relative to the first variant

Needs the authors' code on the path (github.com/EricSteinberger/DREAM: its root for DREAM_and_DeepCFR and PokerRL, plus
no-op stand-ins for gym / pycrayon / scipy.special) and a run of docs/analysis/fhp_reference_run.py for its training
profile.  Results of 2026-10-10: docs/paper-fidelity.md.

    PYTHONPATH=<DREAM>:<stubs>:. python docs/analysis/fhp_fit_authors_network.py <our checkpoint.pt> <reference run dir> <rows> [variant filter,...]
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

from PokerRL.game.games import Flop3Holdem

Flop3Holdem.BIG_BET = 100
from PokerRL.rl import rl_util
from PokerRL.rl.neural.DuelingQNet import DuelingQNet

from headsup.algos.holdem_br import build_street_tree
from headsup.deepcfr.train import loss_weight_scale
from headsup.engine import HeadsUpPoker
from headsup.game import FHP
from headsup.model import BaseModel

dev = "cuda:0"
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
ref_dir, rows = sys.argv[2], int(float(sys.argv[3]))
which = sys.argv[4].split(",") if len(sys.argv) > 4 else None
cfg = dict(ck["model_config"])
T = int(ck["iteration"])
m = ck["adv_memory"][0]
n_all = int(m["size"])
perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))[:rows].sort().values
n = len(perm)
obs = torch.cat([m["obs_int"][perm].float(), m["obs_float"][perm]], 1)
tt, yy = m["t"][perm].clone(), m["target"][perm].clone()
split = torch.randperm(n, generator=torch.Generator().manual_seed(1))
held, train = split[: n // 20], split[n // 20:]
print(f"iteration {T}: {n:,} of {n_all:,} samples of seat 0", flush=True)

# ---- the authors' inputs for our rows: their public observation per decision node, the flop, their range index
with open(os.path.join(ref_dir, "TrainingProfiles", "fhp_es.pkl"), "rb") as f:
    t_prof = pickle.load(f)
env_bldr = rl_util.get_env_builder(t_prof=t_prof)
lut = env_bldr.lut_holder
card_map = torch.as_tensor([int(lut.LUT_2DCARD_2_1DCARD[c % 13, c // 13]) for c in range(52)])
card2d = torch.as_tensor(np.asarray(lut.LUT_1DCARD_2_2DCARD), dtype=torch.long)
hole2idx = torch.as_tensor(np.asarray(lut.LUT_HOLE_CARDS_2_IDX), dtype=torch.long)
bidx = torch.as_tensor(np.asarray(env_bldr.obs_board_idxs), dtype=torch.long)


def hist_hash(stage, hist):  # (rows,), (rows, 48) -> int64 key of the decision node
    h = (hist.double() * 1000).round().long()
    return (h * (torch.arange(1, 49) * 7919)).sum(1) * 4 + stage.long()


def paths_of(nodes, prefix):
    paths = {0: prefix}
    for i, nd in enumerate(nodes):
        for a, c in nd.children.items():
            paths[c] = paths[i] + [a]
    return paths


e = HeadsUpPoker(game=FHP)
e.reset(list(range(9)))
pre_tree = build_street_tree(e)
pp = paths_of(pre_tree, [])
records = [(nd, pp[i]) for i, nd in enumerate(pre_tree) if nd.kind == 0 and nd.stage == 0]
for i, nd in enumerate(pre_tree):
    if nd.kind == 3:
        flop = build_street_tree(nd.engine)
        fp = paths_of(flop, pp[i])
        records += [(fn, fp[j]) for j, fn in enumerate(flop) if fn.kind == 0 and fn.stage == 1]
wrapper = env_bldr.get_new_wrapper(is_evaluating=True)
keys, tmpls, masks, players = [], [], [], []
for nd, path in records:
    wrapper.reset()
    for a in path:
        wrapper.step(a)
    o = torch.as_tensor(nd.engine.observation())
    keys.append(int(hist_hash(o[21:22], o[None, 31:79])[0]))
    tmpls.append(torch.as_tensor(np.asarray(wrapper.get_current_obs(), dtype=np.float32)))
    lm = torch.zeros(3)
    lm[wrapper.env.get_legal_actions()] = 1
    assert lm.bool().tolist() == [bool(v) for v in nd.legal]
    masks.append(lm)
    players.append(nd.player)
order = np.argsort(keys)
skeys = torch.as_tensor(np.asarray(keys)[order])
assert len(set(keys)) == len(keys)
tmpl_t, mask_t = torch.stack(tmpls)[order], torch.stack(masks)[order]
player_t = torch.as_tensor(players)[order]

rk = hist_hash(obs[:, 21], obs[:, 31:79])
node = torch.searchsorted(skeys, rk)
assert bool((skeys[node.clamp(max=len(skeys) - 1)] == rk).all()), "a row's betting history is not a decision node of the tree"
assert bool((player_t[node] == 0).all())  # seat 0's memory
P = tmpl_t[node].clone()
fl_rows = (obs[:, 21] == 1).nonzero().flatten()
theirs = card_map[(obs[fl_rows][:, [8, 11, 14]] - 1).round().long()]
r2 = card2d[theirs]
P[fl_rows[:, None], bidx[None, :]] = torch.stack([r2[..., 0], r2[..., 1], theirs], dim=-1).reshape(len(fl_rows), 9).float()
hc = card_map[(obs[:, [2, 5]] - 1).round().long()]
RIDX = hole2idx[hc.min(1).values, hc.max(1).values].numpy()
LEGAL = mask_t[node]

X, Tt, Y = obs.to(dev), tt.to(dev), yy.to(dev)
P, LEGAL = P.to(dev), LEGAL.to(dev)
tr, he = train.to(dev), held.to(dev)
scale = float(Y[tr[:65536]].pow(2).mean().sqrt())
ws = loss_weight_scale(T, 1.0, "paper")

# ---- pre-flop groups on the training rows
pre = tr[(X[tr, 21] == 0)]
c1, c2 = X[pre, 2].long() - 1, X[pre, 5].long() - 1
r1, rr2, s1, s2 = c1 % 13, c2 % 13, c1 // 13, c2 // 13
cls = torch.maximum(r1, rr2) * 26 + torch.minimum(r1, rr2) * 2 + (s1 == s2).long()
uk, inv = torch.unique(cls * 100 + node.to(dev)[pre], return_inverse=True)
G = len(uk)
first = torch.zeros(G, dtype=torch.long, device=dev).scatter_reduce(0, inv, torch.arange(len(inv), device=dev), "amin", include_self=False)
legal_g = LEGAL[pre][first]
Tp, Yp = Tt[pre], Y[pre]
W = torch.zeros(G, device=dev).index_add_(0, inv, Tp)


def gmean(v):
    return torch.zeros(G, v.shape[1], device=dev).index_add_(0, inv, Tp[:, None] * v) / W[:, None]


tab = gmean(Yp)
se2 = torch.zeros(G, 3, device=dev).index_add_(0, inv, (Tp[:, None] * (Yp - tab[inv])) ** 2) / W[:, None] ** 2
wg = (W / W.sum())[:, None] * legal_g
wg = wg / wg.sum()
rmsg = lambda d: float((wg * d ** 2).sum().sqrt())  # noqa: E731
print(f"pre-flop: {len(pre):,} rows in {G} infosets (class x node); tabular mean regret rms {rmsg(tab):.1f} chips, its standard error "
      f"{float((wg * se2).sum().sqrt()):.1f}; target rms {scale:.1f}", flush=True)
hf = he[(X[he, 21] == 1)][:200_000]
wf = Tt[hf][:, None] * LEGAL[hf]
wf = wf / wf.sum()
rmsf = lambda d: float((wf * d ** 2).sum().sqrt())  # noqa: E731


class Ours(torch.nn.Module):
    def __init__(self, masked):
        super().__init__()
        self.net, self.masked = BaseModel(config=cfg, zero_head=False), masked

    def forward(self, idx):
        return self.net(X[idx])


class Theirs(torch.nn.Module):
    masked = True

    def __init__(self):
        super().__init__()
        self.net = DuelingQNet(q_args=t_prof.module_args["adv_training"].adv_net_args, env_bldr=env_bldr, device=torch.device(dev))

    def forward(self, idx):
        return self.net(pub_obses=P[idx], range_idxs=RIDX[idx.cpu().numpy()], legal_action_masks=LEGAL[idx])


@torch.no_grad()
def predict(model, idx):
    return torch.cat([model(idx[i:i + 65536]) for i in range(0, len(idx), 65536)]) * scale


def fit(make, seed, steps=4000, batch=10000, lr=1e-3, clip=1.0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model = make().to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    clipped = 0
    for step in range(steps):
        idx = tr[torch.randint(0, len(tr), (batch,), device=dev, generator=gen)]
        sq = ws * Tt[idx][:, None] * (model(idx) - Y[idx] / scale).pow(2)
        loss = (sq * LEGAL[idx]).mean() if model.masked else sq.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clipped += float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)) > clip
        opt.step()
    model.eval()
    ph = predict(model, he)
    held_loss = float((Tt[he][:, None] * LEGAL[he] * (ph - Y[he]) ** 2).sum() / (Tt[he][:, None] * LEGAL[he]).sum())
    return dict(pre=gmean(predict(model, pre)), flop=predict(model, hf), held=held_loss, clipped=clipped / steps,
                params=sum(p.numel() for p in model.parameters()))


VARIANTS = {
    "ours (--net deepcfr), loss on all outputs": (lambda: Ours(False), {}),
    "ours, loss on the legal actions": (lambda: Ours(True), {}),
    "authors' network (dueling, masked)": (Theirs, {}),
    "ours, loss on all outputs, 16,000 steps": (lambda: Ours(False), dict(steps=16000)),
    "authors' network, 16,000 steps": (Theirs, dict(steps=16000)),
}
base = None
print(f"{'variant':46s} {'params':>8s} {'noise pre':>9s} {'noise flop':>10s} {'vs tab':>7s} {'held-out':>9s} {'clipped':>8s}")
for name, (make, kw) in VARIANTS.items():
    if which and not any(w in name for w in which):
        continue
    t0 = time.time()
    a, b = fit(make, 0, **kw), fit(make, 1, **kw)
    base = base or a["held"]
    print(f"{name:46s} {a['params']:8,d} {rmsg(a['pre'] - b['pre']) / math.sqrt(2):9.2f} {rmsf(a['flop'] - b['flop']) / math.sqrt(2):10.2f} "
          f"{(rmsg(a['pre'] - tab) + rmsg(b['pre'] - tab)) / 2:7.2f} {(a['held'] + b['held']) / 2 / base:9.4f} {a['clipped']:8.0%}   ({time.time() - t0:.0f}s)", flush=True)
