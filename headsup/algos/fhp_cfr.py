"""Exact tabular Linear CFR on flop hold'em (FHP): the DeepCFR paper's tabular reference (Fig. 3, dashed line).

Full-width, no abstraction, no sampling: every infoset is a (betting history, hand) pre-flop or a (betting
history, flop, hand) on the flop.  Flops are reduced to the 1 755 suit-isomorphism classes - CFR started from the
uniform strategy stays suit-symmetric, so a class representative carries its whole orbit; pre-flop values sum
each representative's hand values over the 24 suit permutations (weighted by orbit size / 24).  Hands are
1326-vectors on the GPU; showdowns use the sign matrices of ``headsup.algos.holdem_br``.  Alternating updates,
regrets and average strategy weighted by t (Linear CFR), regret matching with the uniform fallback - or, with
``--rm-fallback argmax``, the highest-regret action where no regret is positive: Deep CFR's rule, i.e. what Deep CFR
computes with exact regrets instead of sampled ones fitted by a network.  The exploitability of the average strategy
is computed exactly (pre-flop best responses see no flop) and reported like ``headsup.algos.holdem_br``: mean over
seats and total, in mbb/g.

    python -m headsup.algos.fhp_cfr --iterations 50 --eval-at 5,10,20,30,50 --device cuda:0
"""

import argparse
import itertools
import json
import time

import numpy as np
import torch

from headsup import native
from headsup.algos.holdem_br import _HAND_PAIRS, build_street_tree, card_incidence, chance_factor, showdown_signs
from headsup.cards import NUM_CARDS
from headsup.device import get_device
from headsup.engine import HeadsUpPoker
from headsup.game import FHP
from headsup.lbr import COMBOS, NUM_COMBOS

RANKS, SUITS = 13, 4
RM_FALLBACKS = ("uniform", "argmax")


def canonical_flops():
    """(1755, 3) class representatives (sorted card ids, id = rank + 13 suit) and their orbit sizes."""
    perms = list(itertools.permutations(range(SUITS)))
    classes = {}
    for flop in itertools.combinations(range(NUM_CARDS), 3):
        key = min(tuple(sorted(c % RANKS + RANKS * p[c // RANKS] for c in flop)) for p in perms)
        classes[key] = classes.get(key, 0) + 1
    reps = sorted(classes)
    return np.array(reps, dtype=np.int64), np.array([classes[r] for r in reps], dtype=np.float64)


def hand_permutations():
    """(24, 1326): index of the hand sigma(h) for every suit permutation sigma."""
    index = {tuple(sorted(map(int, c))): i for i, c in enumerate(COMBOS)}
    out = []
    for p in itertools.permutations(range(SUITS)):
        m = lambda c: c % RANKS + RANKS * p[c // RANKS]  # noqa: E731
        out.append([index[tuple(sorted((m(int(a)), m(int(b)))))] for a, b in COMBOS])
    return np.array(out, dtype=np.int64)


class FHPCFR:
    def __init__(self, device=None, chunk=64, game=FHP, rm_fallback="uniform"):
        if not (game.limit and game.num_rounds == 2 and not game.all_in):
            raise ValueError("FHPCFR solves two-round limit games without all-ins (FHP): pre-flop + flop only")
        if rm_fallback not in RM_FALLBACKS:
            raise ValueError(f"rm_fallback must be one of {RM_FALLBACKS}, got {rm_fallback!r}")
        self.rm_fallback = rm_fallback
        self.dev = get_device(device) if device is None or isinstance(device, str) else device
        self.chunk = chunk
        e = HeadsUpPoker(game=game)
        e.reset()
        self.game = game
        self.pre = build_street_tree(e)
        self.leaves = [i for i, nd in enumerate(self.pre) if nd.kind == 3]
        self.flop_trees = {i: build_street_tree(self.pre[i].engine) for i in self.leaves}
        flops, w = canonical_flops()
        self.flops = torch.as_tensor(flops, device=self.dev)
        self.flop_w = torch.as_tensor(w, dtype=torch.float32, device=self.dev)
        self.F = len(flops)
        self.hperm = torch.as_tensor(hand_permutations(), device=self.dev)
        incid, disjoint = card_incidence(self.dev)
        self.disjoint_bool = disjoint
        self.disjoint = disjoint.float()
        on = torch.zeros(self.F, NUM_CARDS, device=self.dev)
        on.scatter_(1, self.flops, 1.0)
        self.ok = ((on @ incid.T) == 0).float()  # (F, 1326) hands missing the flop
        strength = np.stack([np.asarray(native.module().BoardTable(list(map(int, f)), 3).strength) for f in flops])
        self.strength = torch.as_tensor(strength.astype(np.int64), device=self.dev)
        self.norm = len(list(itertools.combinations(range(NUM_CARDS), 3))) * chance_factor(0, 3)  # 17 296 flops per hand pair
        A = game.num_actions
        z = lambda *s: torch.zeros(*s, A, device=self.dev)  # noqa: E731
        self.R_pre = {i: z(NUM_COMBOS) for i, nd in enumerate(self.pre) if nd.kind == 0 and nd.stage == 0}
        self.S_pre = {i: z(NUM_COMBOS) for i in self.R_pre}
        self.R_flop = {(l, j): z(self.F, NUM_COMBOS) for l in self.leaves for j, nd in enumerate(self.flop_trees[l]) if nd.kind == 0}
        self.S_flop = {k: z(self.F, NUM_COMBOS) for k in self.R_flop}
        self.iteration = 0

    # -- strategies --------------------------------------------------------------------------------
    @staticmethod
    def _rm(R, legal, fallback="uniform"):
        pos = R.clamp(min=0) * legal
        tot = pos.sum(-1, keepdim=True)
        if fallback == "argmax":  # the highest-regret legal action; exact ties share (all-zero regrets: uniform)
            best = R.masked_fill(legal == 0, float("-inf"))
            top = (best == best.amax(-1, keepdim=True)).to(R.dtype)
            other = top / top.sum(-1, keepdim=True)
        else:
            other = (legal / legal.sum()).expand_as(pos)
        return torch.where(tot > 0, pos / tot.clamp(min=1e-30), other)

    @staticmethod
    def _avg(S, legal):
        tot = (S * legal).sum(-1, keepdim=True)
        uni = legal / legal.sum()
        return torch.where(tot > 0, S * legal / tot.clamp(min=1e-30), uni.expand_as(S))

    def _legal(self, nd):
        return torch.as_tensor(nd.legal, dtype=torch.float32, device=self.dev)

    def _sigma_pre(self, i, average):
        nd = self.pre[i]
        return (self._avg(self.S_pre[i], self._legal(nd)) if average else self._rm(self.R_pre[i], self._legal(nd), self.rm_fallback))

    def _sigma_flop(self, l, j, sl, average):
        nd = self.flop_trees[l][j]
        if average:
            return self._avg(self.S_flop[(l, j)][sl], self._legal(nd))
        return self._rm(self.R_flop[(l, j)][sl], self._legal(nd), self.rm_fallback)

    # -- one street pass ---------------------------------------------------------------------------
    def _flop_values(self, l, sl, reach, p, mode, t, sign):
        """Backward values for player p on the flop tree of pre-flop leaf l, flops ``sl``; reach[s]: (B, 1326).
        mode: 'cfr' (update p's regrets / average with weight t), 'br' (best response to the averages)."""
        nodes = self.flop_trees[l]
        average = mode == "br"
        q = 1 - p
        reaches, sig = {0: reach}, {}
        for j, nd in enumerate(nodes):
            if nd.kind != 0:
                continue
            r = reaches[j]
            s = sig[j] = self._sigma_flop(l, j, sl, average)
            for a, c in nd.children.items():
                child = [r[0], r[1]]
                child[nd.player] = r[nd.player] * s[..., a]
                reaches[c] = child
        vals = {}
        for j in range(len(nodes) - 1, -1, -1):
            nd = nodes[j]
            r = reaches[j]
            if nd.kind == 1:
                vals[j] = (nd.stake if nd.folder != p else -nd.stake) * (r[q] @ self.disjoint)
            elif nd.kind == 2:
                vals[j] = nd.stake * torch.bmm(sign, r[q].unsqueeze(-1)).squeeze(-1)
            elif nd.player == p:
                kids = {a: vals[c] for a, c in nd.children.items()}
                if mode == "br":
                    vals[j] = torch.stack(list(kids.values())).amax(0)
                else:
                    v = sum(sig[j][..., a] * kv for a, kv in kids.items())
                    inst = torch.zeros_like(sig[j])
                    for a, kv in kids.items():
                        inst[..., a] = kv - v
                    self.R_flop[(l, j)][sl] += t * inst * self.ok[sl].unsqueeze(-1)
                    self.S_flop[(l, j)][sl] += t * (r[p] * self.ok[sl]).unsqueeze(-1) * sig[j]
                    vals[j] = v
            else:
                vals[j] = sum(vals[c] for c in nd.children.values())
        return vals[0] * self.ok[sl]

    def _sign(self, sl):
        return showdown_signs(self.strength[sl], self.disjoint_bool)

    def _pass(self, p, mode, t=0.0):
        """One pass for player p over the whole game; returns p's root values (1326)."""
        nodes = self.pre
        average = mode == "br"
        reaches, sig = {0: [torch.ones(NUM_COMBOS, device=self.dev), torch.ones(NUM_COMBOS, device=self.dev)]}, {}
        for i, nd in enumerate(nodes):
            if nd.kind != 0 or nd.stage != 0:
                continue
            r = reaches[i]
            s = sig[i] = self._sigma_pre(i, average)
            for a, c in nd.children.items():
                child = [r[0], r[1]]
                child[nd.player] = r[nd.player] * s[:, a]
                reaches[c] = child
        leaf_vals = {l: torch.zeros(NUM_COMBOS, device=self.dev) for l in self.leaves}
        for c0 in range(0, self.F, self.chunk):
            sl = slice(c0, min(c0 + self.chunk, self.F))
            sign = self._sign(sl)
            for l in self.leaves:
                rl = [x.unsqueeze(0) * self.ok[sl] for x in reaches[l]]
                v = self._flop_values(l, sl, rl, p, mode, t, sign)  # (B, 1326) on the representatives
                v = v[:, self.hperm].sum(1)  # sum over suit permutations: (B, 1326)
                leaf_vals[l] += (self.flop_w[sl, None] / 24.0 * v).sum(0)
        q = 1 - p
        vals = {}
        for i in range(len(nodes) - 1, -1, -1):
            nd = nodes[i]
            if nd.kind == 1:
                vals[i] = (nd.stake if nd.folder != p else -nd.stake) * (reaches[i][q] @ self.disjoint)
            elif nd.kind == 3:
                vals[i] = leaf_vals[i] / self.norm
            elif nd.kind == 0 and nd.stage == 0:
                kids = {a: vals[c] for a, c in nd.children.items()}
                if nd.player != p:
                    vals[i] = sum(kids.values())
                elif mode == "br":
                    vals[i] = torch.stack(list(kids.values())).amax(0)
                else:
                    v = sum(sig[i][:, a] * kv for a, kv in kids.items())
                    inst = torch.zeros_like(sig[i])
                    for a, kv in kids.items():
                        inst[:, a] = kv - v
                    self.R_pre[i] += t * inst
                    self.S_pre[i] += t * reaches[i][p].unsqueeze(-1) * sig[i]
                    vals[i] = v
        return vals[0]

    # -- driver --------------------------------------------------------------------------------------
    def iterate(self):
        self.iteration += 1
        for p in (0, 1):  # alternating updates
            self._pass(p, "cfr", float(self.iteration))

    def exploitability(self):
        br = [float(self._pass(p, "br").sum()) / _HAND_PAIRS for p in (0, 1)]
        expl = 0.5 * (br[0] + br[1])
        return {"exploitability_mbb": 1000.0 * expl / self.game.big_blind, "total_exploitability_mbb": 2000.0 * expl / self.game.big_blind,
                "br_values": br}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--eval-at", default="1,2,5,10,20,30,50")
    p.add_argument("--chunk", type=int, default=64, help="flops per batch")
    p.add_argument("--device", default=None)
    p.add_argument("--rm-fallback", default="uniform", choices=RM_FALLBACKS,
                   help="regret matching without a positive regret: uniform (Linear CFR) | argmax (Deep CFR's rule)")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    cfr = FHPCFR(args.device, args.chunk, rm_fallback=args.rm_fallback)
    print(f"{len(cfr.pre)} pre-flop nodes, {len(cfr.leaves)} flop trees, {len(cfr.R_flop)} flop decision nodes, {cfr.F} flop classes "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    evals = sorted(int(x) for x in args.eval_at.split(","))
    curve = []
    if 0 in evals:
        e = cfr.exploitability()
        curve.append({"iteration": 0, **e})
        print(f"it 0 (uniform): {e['exploitability_mbb']:.1f} mbb/g (total {e['total_exploitability_mbb']:.1f})", flush=True)
    for it in range(1, args.iterations + 1):
        cfr.iterate()
        if it in evals:
            e = cfr.exploitability()
            curve.append({"iteration": it, **e, "seconds": time.perf_counter() - t0})
            print(f"it {it}: average strategy {e['exploitability_mbb']:.1f} mbb/g (total {e['total_exploitability_mbb']:.1f})  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(curve, f, indent=2)


if __name__ == "__main__":
    main()
