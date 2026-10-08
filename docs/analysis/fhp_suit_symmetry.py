"""How suit-symmetric is a trained FHP policy?  For random decision points, apply all 24 suit permutations to the cards
(the game is invariant under them) and compare the network's strategies on the 24 isomorphic infosets."""
import sys, itertools, numpy as np, torch
from headsup.engine import HeadsUpPoker
from headsup.game import FHP
from headsup.players import make_player

PERMS = list(itertools.permutations(range(4)))
def permute(deck, g):
    return [c % 13 + 13 * g[c // 13] for c in deck]

def collect(n, rng, stage):
    """n decision points on the given street: (deck, actions) with a random legal action sequence."""
    out = []
    while len(out) < n:
        deck = [int(c) for c in rng.permutation(52)[:9]]
        e = HeadsUpPoker(game=FHP); e.reset(deck)
        acts = []
        while not e.done:
            if e.stage == stage and rng.random() < 0.4:
                out.append((deck, list(acts)))
                break
            legal = np.flatnonzero(e.legal_mask())
            legal = legal[legal > 0] if len(legal) > 1 else legal  # no folds: stay in the hand
            a = int(rng.choice(legal)); acts.append(a); e.step(a)
    return out

def observations(points):
    obs = []
    for deck, acts in points:
        for g in PERMS:
            e = HeadsUpPoker(game=FHP); e.reset(permute(deck, g))
            for a in acts:
                e.step(a)
            obs.append(e.observation())
    return np.stack(obs).astype(np.float32)

rng = np.random.default_rng(0)
for spec in sys.argv[1:]:
    player = make_player(spec, device="cpu", game=FHP)
    print(spec)
    for stage, name in ((0, "pre-flop"), (1, "flop")):
        pts = collect(1500, rng, stage)
        o = observations(pts)
        p = np.asarray(player.probs(o), dtype=np.float64).reshape(len(pts), 24, -1)
        mean = p.mean(1, keepdims=True)
        l1_to_mean = np.abs(p - mean).sum(-1).mean()
        pair = np.abs(p[:, :, None] - p[:, None, :]).sum(-1)
        l1_pair = pair.sum((1, 2)).mean() / (24 * 23)
        worst = pair.max((1, 2))
        print(f"  {name:8s}: mean L1 distance between the strategies of two isomorphic infosets {l1_pair:.3f} (to the orbit mean {l1_to_mean:.3f}); "
              f"worst pair per infoset: median {np.median(worst):.3f}, 90th percentile {np.quantile(worst, 0.9):.3f}")
