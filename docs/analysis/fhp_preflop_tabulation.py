"""Pre-flop anatomy of a finished FHP run: the samples in its memories recur thousands of times per pre-flop infoset
(169 classes x 4 nodes per seat), so the quantities the networks estimate can be tabulated exactly from the memory:
  - the t-weighted mean sampled regret (what the last regret net should output) with its standard error,
  - the t-weighted mean stored strategy (what the policy net should output).
Compares both with the networks.  Chips; big blind = 100."""
import sys, numpy as np, torch
from headsup.engine import HeadsUpPoker
from headsup.game import FHP
from headsup.players import make_player, regret_matching_torch
from headsup.sdcfr import IterateBank

torch.set_num_threads(8)
run = sys.argv[1]
ck = torch.load(f"{run}/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
T = int(ck["iteration"])
bank = IterateBank.load(f"{run}/iterates.pt", "cpu")
policy = make_player(f"cfr:{run}/policy.pth", device="cpu", game=FHP)
SEQS = ["", "c", "cr", "crr", "crrr", "r", "rr", "rrr"]

def cls_of(c1, c2):
    r1, r2, s1, s2 = c1 % 13, c2 % 13, c1 // 13, c2 // 13
    return max(r1, r2) * 26 + min(r1, r2) * 2 + int(s1 == s2)

# every combo at every pre-flop node: full observation, class, node, legal mask
rows = []
for ni, seq in enumerate(SEQS):
    seat = len(seq) % 2
    for a in range(52):
        for b in range(a + 1, 52):
            rest = [c for c in range(52) if c not in (a, b)]
            deck = ([a, b] + rest[:2] if seat == 0 else rest[:2] + [a, b]) + rest[2:7]
            e = HeadsUpPoker(game=FHP); e.reset(deck)
            for ch in seq:
                e.step(1 if ch == "c" else 2)
            rows.append((e.observation(), cls_of(a, b), ni, seat, np.asarray(e.legal_mask(), bool)))
OBS = np.stack([r[0] for r in rows]).astype(np.float32)
CLS = np.array([r[1] for r in rows]); NODE = np.array([r[2] for r in rows]); SEAT = np.array([r[3] for r in rows])
LEGAL = np.stack([r[4] for r in rows])
hist_key = lambda h: (np.round(np.asarray(h, np.float64) * 1000).astype(np.int64) * (np.arange(1, 13) * 7919)).sum(-1)
node_of_key = {int(hist_key(OBS[NODE == ni][0, 31:43])): ni for ni in range(8)}
assert len(node_of_key) == 8
gid = CLS * 8 + NODE                                   # group = class x node
groups, ginv = np.unique(gid, return_inverse=True)
G = len(groups)
legal_g = np.zeros((G, 3), bool); legal_g[ginv] = LEGAL
seat_g = np.zeros(G, int); seat_g[ginv] = SEAT
def class_mean(v):                                      # mean over a group's combos (rows of the table)
    out = np.zeros((G, v.shape[1])); np.add.at(out, ginv, v)
    return out / np.bincount(ginv, minlength=G)[:, None]

def memory_groups(mem):
    """pre-flop rows of a memory -> (group index per row, t, target)"""
    n = int(mem["size"])
    oi = mem["obs_int"][:n]
    pre = (oi[:, 21] == 0).nonzero().flatten()
    c1, c2 = oi[pre, 2].long().numpy() - 1, oi[pre, 5].long().numpy() - 1
    r1, r2, s1, s2 = c1 % 13, c2 % 13, c1 // 13, c2 // 13
    cls = np.maximum(r1, r2) * 26 + np.minimum(r1, r2) * 2 + (s1 == s2)
    hist = mem["obs_float"][pre][:, 31 - 23:43 - 23].numpy()
    keys = hist_key(hist)
    node = np.vectorize(node_of_key.get)(keys)
    g = np.searchsorted(groups, cls * 8 + node)
    return g, mem["t"][pre].numpy().astype(np.float64), mem["target"][pre].numpy().astype(np.float64), len(pre), n

def wstats(g, t, y):
    W = np.bincount(g, weights=t, minlength=G)
    mean = np.stack([np.bincount(g, weights=t * y[:, a], minlength=G) for a in range(3)], 1) / np.maximum(W, 1e-12)[:, None]
    var = np.stack([np.bincount(g, weights=(t * (y[:, a] - mean[g, a])) ** 2, minlength=G) for a in range(3)], 1) / np.maximum(W, 1e-12)[:, None] ** 2
    return W, mean, np.sqrt(var)

rm = lambda adv, legal: regret_matching_torch(torch.as_tensor(np.where(legal, adv, -1e9), dtype=torch.float32), torch.as_tensor(legal), bank.rm_fallback).numpy().astype(np.float64)
print(f"{run}: iteration {T}")
for seat in (0, 1):
    g, t, y, npre, n = memory_groups(ck["adv_memory"][seat])
    W, tab, se = wstats(g, t, y)
    sel = (seat_g == seat) & (W > 0)
    w = (W / W[sel].sum())[:, None] * legal_g * sel[:, None]
    rms = lambda d: float(np.sqrt((w * d ** 2).sum() / w.sum()))
    x = torch.as_tensor(OBS[:, : bank.obs_dim])
    with torch.no_grad():
        net = bank._fn({k: v[-1] for k, v in bank.params[seat].items()}, x).numpy().astype(np.float64)
        prev = bank._fn({k: v[-2] for k, v in bank.params[seat].items()}, x).numpy().astype(np.float64)
    netg, prevg = class_mean(net), class_mean(prev)
    s_tab, s_net, s_prev = rm(tab, legal_g), rm(netg, legal_g), rm(prevg, legal_g)
    wi = W * sel / (W * sel).sum()
    l1 = lambda a, b: float((wi * np.abs(a - b).sum(1)).sum())
    top = np.sort(np.where(legal_g, tab, -np.inf), 1)
    margin = top[:, -1] - top[:, -2]
    q = lambda v, p: float(np.interp(p, np.cumsum(wi[np.argsort(v)]), np.sort(v)))
    print(f" seat {seat}: {npre:,} pre-flop samples of {n:,} in {int(sel.sum())} infosets")
    print(f"   mean regret (tabulated) rms {rms(tab):.1f}, its standard error {rms(se):.1f}; last net against it: rms {rms(netg - tab):.1f}; "
          f"last two nets against each other: {rms(netg - prevg):.1f}")
    print(f"   regret gap between the two best actions (tabulated): quartiles {q(margin, .25):.1f} / {q(margin, .5):.1f} / {q(margin, .75):.1f} chips; "
          f"share of infosets (by visits) with a gap below 5 / 10 / 20 chips: {float(wi[margin < 5].sum()):.2f} / {float(wi[margin < 10].sum()):.2f} / {float(wi[margin < 20].sum()):.2f}")
    print(f"   strategies by regret matching: L1(table, last net) {l1(s_tab, s_net):.3f}; L1(last net, the one before) {l1(s_net, s_prev):.3f}; "
          f"share where the table and the net choose different most-likely actions {float(wi[s_tab.argmax(1) != s_net.argmax(1)].sum()):.2f}")
# the average strategy: tabulated from the strategy memory against the policy net
g, t, y, npre, n = memory_groups(ck["strat_memory"])
W, tab, se = wstats(g, t, y)
pol = class_mean(np.asarray(policy.probs(OBS), np.float64))
for seat in (0, 1):
    sel = (seat_g == seat) & (W > 0)
    wi = W * sel / (W * sel).sum()
    print(f" average strategy, seat {seat}: L1(tabulated from the strategy memory, policy net) {float((wi * np.abs(tab - pol).sum(1)).sum()):.3f} "
          f"(standard error of the tabulated probabilities: {float(np.sqrt((wi[:, None] * se ** 2).sum() / 3)):.4f})")
np.savez(f"{run}/preflop_tab.npz", groups=groups, tab_strategy=tab, weight=W, legal=legal_g, seat=seat_g)
