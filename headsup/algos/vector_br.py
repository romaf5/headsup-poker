"""Best response and exploitability of hold'em strategies, vectorised over hands and boards (torch).

The best response is exact over all 1326 hands and betting sequences.  Chance is handled street
by street: the cards of the next street are enumerated (``cards="all"``; FHP's 22 100 flops) or
sampled per public state (``cards=k``), and a street's values are averaged over its chance
outcomes *before* the responder maximises at the previous street, so the responder never sees
cards it could not know (unbiased street expectations; the maximum over sampled estimates is
biased upwards, the bias vanishing as the number of cards grows - enumerate when possible).

Players are mixtures of T behaviour strategies with weights w_t: T = 1 for a policy or iterate
network, the SD-CFR average is the w_t = t^gamma mixture of its iterates.  A mixture's realisation
weight of a sequence is the weighted mean of its components' realisation weights, which is all a
best response needs (the opponent's reach vectors), so the SD-CFR average strategy is evaluated
exactly, without forming its behaviour strategy.

Chance normalisation: values of a street are summed over its card combinations with the hands
overlapping the cards masked and divided by ``N * P(the new cards miss a fixed pair of hands)``
(the same for every hand pair), an unbiased estimate of the expectation given the hands.

    python -m headsup.algos.vector_br --policy cfr:runs/fhp/policy.pth --cards all
    python -m headsup.algos.vector_br --policy sdcfr:runs/fhp/iterates.pt --cards 2000
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from headsup import native
from headsup.algos.holdem_br import build_street_tree
from headsup.cards import CARD_FEATURES, NUM_CARDS
from headsup.engine import BOARD_CARDS_BY_STAGE, HeadsUpPoker
from headsup.lbr import COMBO_FEATURES, COMBOS, NUM_COMBOS

_HAND_PAIRS = NUM_COMBOS * 1225  # hand pairs without a shared card (C(50, 2) opponent hands per hand)


def _compat_factor(before, after):
    """P(the ``after - before`` newly dealt cards miss a fixed pair of hands (4 cards) | ``before`` known)."""
    f = 1.0
    for i in range(before, after):
        f *= (NUM_CARDS - 4 - i) / (NUM_CARDS - i)
    return f


# ----------------------------------------------------------------------------------------- policies
class MixturePolicy:
    """T behaviour strategies with weights: ``strategies(seat, x, legal) -> (T, R, A)`` for observation
    rows ``x`` (R, obs) on ``device`` and the node's legal mask (A,) bool."""

    def __init__(self, fn, weights, game, device, max_rows=1 << 17):
        self.fn, self.game, self.device = fn, game, torch.device(device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        self.T = len(self.weights)
        self.max_rows = max_rows

    @torch.no_grad()
    def strategies(self, seat, x, legal):
        outs = [self.fn(seat, x[i : i + self.max_rows], legal) for i in range(0, len(x), self.max_rows)]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def _masked_softmax(logits, legal):
    p = torch.softmax(logits.float(), dim=-1) * legal
    s = p.sum(-1, keepdim=True)
    uniform = legal.float() / legal.sum()
    return torch.where(s > 0, p / s.clamp(min=1e-30), uniform)


def mixture_policy(player, device):
    """Wrap a network player (policy net, iterate, SD-CFR bank) as a :class:`MixturePolicy`."""
    from headsup.players import RegretMatchingPlayer, TorchPolicyPlayer, regret_matching_torch
    from headsup.sdcfr import SDCFRPlayer

    device = torch.device(device)
    if isinstance(player, TorchPolicyPlayer):
        model = player.model.to(device).eval()
        return MixturePolicy(lambda seat, x, legal: _masked_softmax(model(x), legal)[None], [1.0], player.game, device)
    if isinstance(player, RegretMatchingPlayer):
        nets = [n.to(device).eval() for n in player.nets]
        fb = getattr(nets[0], "rm_fallback", "uniform")
        return MixturePolicy(lambda seat, x, legal: regret_matching_torch(nets[seat](x), legal.expand(len(x), -1), fb)[None],
                             [1.0], player.game, device)
    if isinstance(player, SDCFRPlayer):
        bank = player.bank
        if bank.device != device:
            raise ValueError("load the SD-CFR bank on the evaluation device")
        rows = max(256, bank.ACTIVATION_BUDGET // (bank.T * bank.dim * 4 * 12))

        def fn(seat, x, legal):
            outs = []
            for i in range(0, len(x), rows):
                xi = x[i : i + rows, : bank.obs_dim]
                adv = bank._vmapped(bank.params[seat], xi)
                outs.append(regret_matching_torch(adv, legal.expand(len(xi), -1), bank.rm_fallback))
            return torch.cat(outs, dim=1)

        return MixturePolicy(fn, bank.weights, player.game, device)
    raise TypeError(f"no vectorised strategy for {type(player).__name__} (network players only)")


# ----------------------------------------------------------------------------------------- the solver
class VectorBestResponse:
    def __init__(self, policy, game, cards="all", chunk=64, seed=0):
        """``cards``: "all" to enumerate every street's cards, or k = cards sampled per public state
        and street (the flop: k flops; turn / river: k cards each, nested); ``chunk``: boards per batch."""
        self.policy, self.game = policy, game
        self.dev = policy.device
        self.cards = cards
        self.chunk = chunk
        self.rng = np.random.default_rng(seed)
        self.final_cards = BOARD_CARDS_BY_STAGE[game.num_rounds - 1]
        self.combo_feat = torch.as_tensor(COMBO_FEATURES, device=self.dev)
        self.card_feat = torch.as_tensor(CARD_FEATURES, device=self.dev)
        combos = torch.as_tensor(COMBOS, device=self.dev)
        incid = torch.zeros(NUM_COMBOS, NUM_CARDS, device=self.dev)
        incid[torch.arange(NUM_COMBOS), combos[:, 0]] = 1.0
        incid[torch.arange(NUM_COMBOS), combos[:, 1]] = 1.0
        self.incid = incid  # (1326, 52) card incidence
        self.disjoint = (incid @ incid.T) == 0  # (1326, 1326) hands without a shared card
        self.trees = {}
        self.queries = 0
        self._sd_boards = self._sd_sign = None

    # -- helpers ---------------------------------------------------------------------------------
    def _tree(self, engine):
        key = (tuple(engine.bets), tuple(engine.stage_bets), int(engine.stage), int(engine.current),
               tuple(tuple(x) for x in engine.history_size), tuple(engine.history_n), engine.consecutive_raises)
        t = self.trees.get(key)
        if t is None:
            t = self.trees[key] = build_street_tree(engine)
        return t

    def _ok(self, boards):
        """(B, 1326) hands that miss every card of ``boards`` (B, n)."""
        if boards.shape[1] == 0:
            return torch.ones(len(boards), NUM_COMBOS, dtype=torch.bool, device=self.dev)
        on_board = torch.zeros(len(boards), NUM_CARDS, device=self.dev)
        on_board.scatter_(1, boards, 1.0)
        return (on_board @ self.incid.T) == 0

    def _obs(self, engine, seat, boards):
        """(B * 1326, obs) observation rows of ``seat`` at the node for every board and hand."""
        base = torch.as_tensor(engine.observation(seat), device=self.dev)
        B, n = boards.shape
        x = base.expand(B, NUM_COMBOS, -1).clone()
        x[:, :, :6] = self.combo_feat
        if n:
            flop = torch.sort(boards[:, :3], dim=1).values
            cards = torch.cat([flop, boards[:, 3:]], dim=1)
            x[:, :, 6 : 6 + 3 * n] = self.card_feat[cards].reshape(B, 1, 3 * n)
        return x.reshape(B * NUM_COMBOS, -1)

    def _strengths(self, boards):
        """(B, 1326) showdown strengths (treys: lower is stronger) on the final boards."""
        b = boards.cpu().numpy()
        out = np.stack([np.asarray(native.module().BoardTable(list(map(int, row)), self.final_cards).strength) for row in b])
        return torch.as_tensor(out.astype(np.int64), device=self.dev)

    def _next_cards(self, boards, k):
        """Child boards of each board (dealing the next street's cards) and the parent index."""
        B, n = boards.shape
        m = BOARD_CARDS_BY_STAGE[BOARD_CARDS_BY_STAGE.index(n) + 1] - n
        b = boards.cpu().numpy()
        kids, parent = [], []
        for i in range(B):
            rest = np.setdiff1d(np.arange(NUM_CARDS), b[i])
            if k == "all":
                import itertools

                deals = np.array(list(itertools.combinations(rest, m)), dtype=np.int64).reshape(-1, m)
            else:
                deals = np.stack([np.sort(self.rng.choice(rest, m, replace=False)) for _ in range(int(k))])
            kids.append(np.concatenate([np.repeat(b[i][None], len(deals), 0), deals], axis=1))
            parent.append(np.full(len(deals), i))
        return torch.as_tensor(np.concatenate(kids), device=self.dev), torch.as_tensor(np.concatenate(parent), device=self.dev)

    # -- one street ------------------------------------------------------------------------------
    def street(self, engine, boards, rho):
        """Values of both players' hands at the street root ``engine`` for each board in ``boards``
        (B, n visible cards), given the per-component own-reach tensors ``rho[s]`` (T, B, 1326) at the
        root (zero for hands blocked by the board).  Returns {p: (br (B, 1326), onpolicy (B, 1326))}:
        counterfactual values (weighted by the opponent's realisation weights, summed over its hands)."""
        nodes = self._tree(engine)
        w = self.policy.weights
        W = w.sum()
        B = len(boards)
        real = [dict(), dict()]  # realisation weight per node: (B, 1326) per seat
        pending = {0: rho}
        leaf_values = {}
        for i, nd in enumerate(nodes):
            r = pending.pop(i)
            for s in (0, 1):
                real[s][i] = torch.einsum("t,tbh->bh", w, r[s]) / W
            if nd.kind == 0 and nd.stage == nodes[0].stage:
                seat = nd.player
                legal = torch.as_tensor(nd.legal, device=self.dev)
                sig = self.policy.strategies(seat, self._obs(nd.engine, seat, boards), legal)
                self.queries += 1
                sig = sig.reshape(self.policy.T, B, NUM_COMBOS, -1)
                for a, c in nd.children.items():
                    child = [r[0], r[1]]
                    child[seat] = r[seat] * sig[..., a]
                    pending[c] = child
            elif nd.kind == 3:
                leaf_values[i] = self._next_street(nd.engine, boards, r)
        showdown_cache = {}
        out = {}
        for p in (0, 1):
            q = 1 - p
            br, onp = {}, {}
            for i in range(len(nodes) - 1, -1, -1):
                nd = nodes[i]
                if nd.kind == 1:
                    mass = self._opp_mass(real[q][i])
                    v = (nd.stake if nd.folder != p else -nd.stake) * mass
                    br[i] = onp[i] = v
                elif nd.kind == 2:
                    key = (i, q)
                    if key not in showdown_cache:
                        showdown_cache[key] = self._showdown(boards, real[q][i])
                    br[i] = onp[i] = nd.stake * showdown_cache[key]
                elif nd.kind == 3:
                    br[i], onp[i] = leaf_values[i][p]
                else:
                    kids = list(nd.children.values())
                    if nd.player == p:
                        br[i] = torch.stack([br[c] for c in kids]).amax(0)
                        rp = real[p][i]
                        onp[i] = sum(torch.where(rp > 0, real[p][c] / rp.clamp(min=1e-30), 0.0) * onp[c] for c in kids)
                    else:
                        br[i] = sum(br[c] for c in kids)
                        onp[i] = sum(onp[c] for c in kids)
            out[p] = (br[0], onp[0])
        return out

    def _opp_mass(self, reach):
        """sum of the opponent's realisation weights over hands without a card of h: (B, 1326)."""
        return reach @ self.disjoint.float()

    def _showdown(self, boards, reach):
        """u[b, h] = sum_h' reach[b, h'] * (+1 if h beats h', -1 if it loses) over compatible h'."""
        if boards.shape[1] != self.final_cards:
            raise NotImplementedError("showdowns before the last street (all-ins) are not supported yet")
        if self._sd_boards is not boards:
            s = self._strengths(boards).clamp(max=1 << 20).to(torch.int32)  # blocked hands: INT32_MAX -> 2^20
            self._sd_sign = torch.sign((s[:, None, :] - s[:, :, None]).float()) * self.disjoint  # (B, h, h')
            self._sd_boards = boards
        return torch.bmm(self._sd_sign, reach.unsqueeze(-1)).squeeze(-1)

    def _next_street(self, engine, boards, rho):
        """Values at a street-end leaf (``engine`` = the next street's root): the next street's values
        for every dealt card combination, masked by the hands' compatibility, averaged per parent board."""
        n, n_next = boards.shape[1], BOARD_CARDS_BY_STAGE[int(engine.stage)]
        kids, parent = self._next_cards(boards, self.cards)
        per_parent = len(kids) // len(boards)
        acc = {p: [torch.zeros(len(boards), NUM_COMBOS, device=self.dev) for _ in range(2)] for p in (0, 1)}
        for c0 in range(0, len(kids), self.chunk):
            kb, kp = kids[c0 : c0 + self.chunk], parent[c0 : c0 + self.chunk]
            ok = self._ok(kb).float()
            vals = self.street(engine_with_board(engine, kb), kb, [x[:, kp] * ok for x in rho])
            for p in (0, 1):
                for j in (0, 1):
                    acc[p][j].index_add_(0, kp, vals[p][j] * ok)
        norm = per_parent * _compat_factor(n, n_next)
        return {p: (acc[p][0] / norm, acc[p][1] / norm) for p in (0, 1)}

    # -- driver ----------------------------------------------------------------------------------
    def run(self):
        e = HeadsUpPoker(game=self.game)
        e.reset()
        boards = torch.zeros(1, 0, dtype=torch.long, device=self.dev)
        ones = torch.ones(self.policy.T, 1, NUM_COMBOS, device=self.dev)
        vals = self.street(e, boards, [ones, ones.clone()])
        br = [float(vals[p][0].sum()) / _HAND_PAIRS for p in (0, 1)]
        onp = [float(vals[p][1].sum()) / _HAND_PAIRS for p in (0, 1)]
        expl = 0.5 * (br[0] + br[1])
        return {"exploitability_chips": expl, "exploitability_mbb": 1000.0 * expl / self.game.big_blind,
                "total_exploitability_mbb": 2000.0 * expl / self.game.big_blind,
                "br_values": br, "values": onp, "cards": self.cards, "strategy_queries": self.queries}


def engine_with_board(engine, boards):
    """The street-root engine with the first board of ``boards`` visible (the public tree is the same
    for every board; observations are patched per board)."""
    e = engine.clone()
    b = [int(c) for c in boards[0].tolist()]
    rest = [c for c in range(NUM_CARDS) if c not in b]
    e.board = tuple(b + rest[: 5 - len(b)])
    e.hands = ((rest[-1], rest[-2]), (rest[-3], rest[-4]))
    return e


def main(argv=None):
    from headsup.games.holdem import make_holdem
    from headsup.players import make_player

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="network player spec: cfr:..., iterate:..., sdcfr:... (exact average)")
    p.add_argument("--game", default="fhp", choices=["fhp", "hulh", "nlhe"], help="for players without their own action tree")
    p.add_argument("--cards", default="all", help="'all' (enumerate) or k cards sampled per public state and street")
    p.add_argument("--chunk", type=int, default=64, help="boards per batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    from headsup.device import get_device

    device = get_device(args.device)
    player = make_player(args.policy, device=device, seed=args.seed, game=make_holdem(args.game))
    game = player.game
    cards = args.cards if args.cards == "all" else int(args.cards)
    t0 = time.perf_counter()
    res = VectorBestResponse(mixture_policy(player, device), game, cards=cards, chunk=args.chunk, seed=args.seed).run()
    res["seconds"] = time.perf_counter() - t0
    res["policy"] = args.policy
    print(f"{args.policy}: exploitability {res['exploitability_mbb']:.1f} mbb/g (total {res['total_exploitability_mbb']:.1f}; "
          f"{res['exploitability_chips']:.3f} chips; BR values {res['br_values'][0]:+.3f} / {res['br_values'][1]:+.3f}, "
          f"profile values {res['values'][0]:+.3f} / {res['values'][1]:+.3f}; cards {cards}) in {res['seconds']:.0f}s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
