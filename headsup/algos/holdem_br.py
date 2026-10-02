"""Exploitability of hold'em strategies: best response exact over hands, vectorised over boards (torch).

``VectorBestResponse`` (the CLI) is exact over all 1326 hands and betting sequences.  Chance is
handled street by street: the next street's cards are enumerated (``cards="all"``; FHP's 22 100
flops) or sampled per public state (``cards=k`` or one k per street, nested), and a street's
values are averaged over its cards *before* the responder maximises at the previous street, so it
never sees cards it could not know.  Averages over enumerated cards are exact; sampled ones are
unbiased, and the maximum over them is biased upwards (the bias vanishes as k grows).  Values
summed over card combinations mask the hands that overlap the cards and are divided by
``N * P(the new cards miss a fixed pair of hands)`` - the same for every pair - which makes them
the expectations given the hands.  All-in showdowns before the last street average the equity
matrix over the rest of the board (enumerated from the flop on, sampled pre-flop).

Players are mixtures of T behaviour strategies with weights w_t: T = 1 for any stateless player
(policy / iterate networks on the GPU, other players through ``probs``), the SD-CFR average is the
w_t = t^gamma mixture of its iterates.  A mixture's realisation weight of a sequence is the weighted
mean of its components' realisation weights - all a best response needs (the opponent's reach
vectors) - so the SD-CFR average is evaluated exactly, without forming its behaviour strategy.

Units: ``exploitability`` = (BR_0 + BR_1) / 2 (the mean over seats, as in most papers and our
Leduc numbers); the DeepCFR paper's FHP figures are *total* exploitability BR_0 + BR_1
(``total_exploitability_mbb``).

``HoldemBestResponse`` is the per-board reference implementation (numpy, one board at a time; a
sampled full board is followed through the streets, so on multi-street subgames the responder
sees the later cards): used to validate subgame solvers on single boards.

    python -m headsup.algos.holdem_br --policy cfr:runs/fhp/policy.pth --cards all
    python -m headsup.algos.holdem_br --policy sdcfr:runs/fhp/iterates.pt --cards 2000
    python -m headsup.algos.holdem_br --policy tab:models/blueprint_nlhe.pt --cards 200,10,10
"""

import argparse
import itertools
import json
import time

import numpy as np
import torch

from headsup import native
from headsup.cards import CARD_FEATURES, NUM_CARDS, hand_strength
from headsup.engine import BOARD_CARDS_BY_STAGE, HeadsUpPoker
from headsup.game import FHP, GameConfig
from headsup.lbr import COMBO_FEATURES, COMBOS, NUM_COMBOS, substitute_hands, valid_combos
from headsup.search import _opponent_mass, _showdown_values

_HAND_PAIRS = NUM_COMBOS * 1225  # hand pairs without a shared card (C(50, 2) opponent hands per hand)


class _Node:
    __slots__ = ("engine", "kind", "player", "folder", "stake", "children", "legal", "twins", "stage")

    def __init__(self, engine):
        self.engine = engine
        self.stage = int(engine.stage)
        if engine.done:
            self.kind = 1 if engine.folded >= 0 else 2
            self.player = -1
            self.folder = engine.folded
            self.stake = engine.bets[engine.folded] if engine.folded >= 0 else min(engine.bets)
            self.children = {}
            self.legal = self.twins = None
        else:
            self.kind = 0
            self.player = engine.current
            self.folder, self.stake = -1, 0
            self.legal, self.twins = engine.legal_mask_and_twins()
            self.children = {}


def build_street_tree(engine):
    """Public tree of the current street: dict id -> _Node; children of decision nodes per legal
    action, street-end leaves are decision nodes of the *next* street (kind 3)."""
    nodes = [_Node(engine)]
    i = 0
    while i < len(nodes):
        nd = nodes[i]
        if nd.kind == 0 and nd.stage == nodes[0].stage:
            for a in np.flatnonzero(nd.legal):
                c = nd.engine.clone()
                c.step(int(a))
                child = _Node(c)
                if child.kind == 0 and child.stage != nodes[0].stage:
                    child.kind = 3  # street-end leaf
                nd.children[int(a)] = len(nodes)
                nodes.append(child)
        i += 1
    return nodes


def chance_factor(revealed_before, revealed_after):
    """P(the board cards revealed between the two counts are compatible with a fixed pair of hands)
    when the board is drawn uniformly: the same for every hand pair (4 blocked cards), which is
    what makes ``sum over sampled boards / (N * factor)`` an unbiased, exactly zero-sum estimate."""
    f = 1.0
    for i in range(revealed_before, revealed_after):
        f *= (NUM_CARDS - 4 - i) / (NUM_CARDS - i)
    return f


class _NativeShowdown:
    def __init__(self, board):
        self.table = native.module().BoardTable(board, len(board))

    def __call__(self, reach):
        return self.table.showdown_values(reach)


class _NumpyShowdown:
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, reach):
        return _showdown_values(reach, self.strength)


class HoldemBestResponse:
    def __init__(self, player, game=FHP, boards=200, seed=0, batch=8192):
        self.player, self.game = player, game
        self.rng = np.random.default_rng(seed)
        self.n_boards = boards
        self.batch = batch
        self.probs_cache = {}
        self.calls = 0

    # -- strategy queries -------------------------------------------------------------------
    def _strategy(self, node, seat):
        """(1326, A) strategy of ``seat`` at the public node (its observation with every hand)."""
        e = node.engine
        key = (seat, tuple(e.visible_board), tuple(e.bets), tuple(e.stage_bets), int(e.stage), e.consecutive_raises,
               tuple(tuple(x) for x in e.history_size), tuple(e.history_n))
        cached = self.probs_cache.get(key)
        if cached is not None:
            return cached
        rows = substitute_hands(e.observation(seat))
        probs = np.asarray(self.player.probs(rows), dtype=np.float64)
        self.calls += 1
        self.probs_cache[key] = probs
        return probs

    # -- values -------------------------------------------------------------------------------
    def _values(self, nodes, node_id, p, reach_q, boards, strength, best_response):
        """Counterfactual value vector of player p's hands at ``node_id`` given the opponent's
        reach vector; ``boards`` = the sampled boards this subtree may see (list of card tuples),
        ``strength`` = per-board strength arrays for the showdown."""
        nd = nodes[node_id]
        if nd.kind == 1:
            sign = 1.0 if nd.folder != p else -1.0
            return sign * nd.stake * _opponent_mass(reach_q)
        n_final = BOARD_CARDS_BY_STAGE[self.game.num_rounds - 1]
        revealed = len(nodes[0].engine.visible_board)  # cards known on this street (the node's own engine may be past it)
        if nd.kind == 2:
            # showdown on the boards' final cards (the unrevealed ones are dealt): sum over the sampled
            # boards, hands overlapping a board contribute nothing there, normalised by N * P(compatible)
            out = np.zeros(NUM_COMBOS)
            for b, s in zip(boards, strength):
                ok = valid_combos(b[:n_final])
                out += nd.stake * s(np.where(ok, reach_q, 0.0)) * ok
            return out / (len(boards) * chance_factor(revealed, n_final))
        if nd.kind == 3:
            # street end: continue on each sampled board with its next street's cards
            out = np.zeros(NUM_COMBOS)
            nxt = None
            for b, s in zip(boards, strength):
                sub = self._street_from(nd.engine, b)
                nxt = BOARD_CARDS_BY_STAGE[int(sub[0].engine.stage)]
                ok = valid_combos(b[:nxt])
                out += self._values(sub, 0, p, np.where(ok, reach_q, 0.0), [b], [s], best_response) * ok
            return out / (len(boards) * chance_factor(revealed, nxt))
        acts = [a for a in range(self.game.num_actions) if nd.legal[a]]
        if nd.player == p:
            child_values = np.stack([self._values(nodes, nd.children[a], p, reach_q, boards, strength, best_response) for a in acts])
            if best_response:
                return child_values.max(axis=0)
            sig = self._strategy(nd, p)
            return (sig[:, acts].T * child_values).sum(axis=0)
        sig = self._strategy(nd, nd.player)
        return sum(self._values(nodes, nd.children[a], p, reach_q * sig[:, a], boards, strength, best_response) for a in acts)

    def _street_from(self, engine, board):
        """Public tree of the next street after ``engine``'s street-end state on ``board``."""
        e = engine.clone()
        e.board = tuple(board)  # the sampled board (the engine's dummy cards are replaced)
        spare = [c for c in range(NUM_CARDS) if c not in board][:4]  # dummy hands off the board (showdowns evaluate them)
        e.hands = ((spare[0], spare[1]), (spare[2], spare[3]))
        return build_street_tree(e)

    def _strengths(self, boards):
        """Per board: the showdown-payoff evaluator (the C++ table when the extension is built,
        otherwise the numpy strength-sorted cumulative sums of headsup.search)."""
        n_final = BOARD_CARDS_BY_STAGE[self.game.num_rounds - 1]
        if native.available():
            return [_NativeShowdown(list(b[:n_final])) for b in boards]
        out = []
        for b in boards:
            s = np.full(NUM_COMBOS, np.inf)
            final = list(b[:n_final])
            for h in np.flatnonzero(valid_combos(final)):
                s[h] = hand_strength([int(COMBOS[h, 0]), int(COMBOS[h, 1])], final)
            out.append(_NumpyShowdown(s))
        return out

    def evaluate_from(self, engine, ranges, boards):
        """Best responses from a given public state with given ranges over the given full boards
        (a list of 5-card tuples whose known prefix must match the engine's board)."""
        root = build_street_tree(engine)
        strength = self._strengths(boards)
        r = [np.asarray(x, dtype=np.float64) for x in ranges]
        joint = float(_opponent_mass(r[1]) @ r[0])
        out = {}
        for p in (0, 1):
            br = float(r[p] @ self._values(root, 0, p, r[1 - p], boards, strength, True)) / joint
            v = float(r[p] @ self._values(root, 0, p, r[1 - p], boards, strength, False)) / joint
            out[p] = (br, v)
        expl = 0.5 * (out[0][0] + out[1][0])
        return {"exploitability_chips": expl, "exploitability_mbb": 1000.0 * expl / self.game.big_blind,
                "br_values": [out[0][0], out[1][0]], "values": [out[0][1], out[1][1]], "boards": len(boards), "strategy_queries": self.calls}

    def run(self):
        e = HeadsUpPoker(rng=self.rng, game=self.game)
        e.reset()
        boards = [tuple(int(c) for c in self.rng.permutation(NUM_CARDS)[:5]) for _ in range(self.n_boards)]
        r = np.ones(NUM_COMBOS)  # uniform ranges (the deal)
        return self.evaluate_from(e, [r, r], boards)


# ----------------------------------------------------------------------------------------- policies
class MixturePolicy:
    """T behaviour strategies with weights: ``strategies(seat, x, legal) -> (T, R, A)`` for observation
    rows ``x`` (R, obs) on ``device`` and the node's legal mask (A,) bool."""

    def __init__(self, fn, weights, game, device, max_rows=1 << 17):
        """``fn(seat, x (R, obs), legal (A,) bool) -> (T, R, A)`` behaviour strategies, ``weights`` (T,)."""
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


def mixture_policy(player, device, game=None):
    """Wrap a player as a :class:`MixturePolicy`: network players (policy net, iterate, SD-CFR bank)
    run on ``device``; any other stateless player is queried through ``probs`` (T = 1)."""
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
    if getattr(player, "wants_ids", False):
        raise TypeError(f"{type(player).__name__} is stateful: its strategy is not a function of the observation")
    game = getattr(player, "game", None) or game

    def generic(seat, x, legal):
        p = torch.as_tensor(np.asarray(player.probs(x.cpu().numpy()), dtype=np.float32), device=x.device) * legal
        return (p / p.sum(-1, keepdim=True).clamp(min=1e-30))[None]

    return MixturePolicy(generic, [1.0], game, device, max_rows=1 << 20)


# ----------------------------------------------------------------------------------------- the solver
def parse_cards(text):
    """'all' | 'k' | 'k1,k2,k3' (per street: flops, turns per flop, rivers per turn; 'all' allowed)."""
    parts = [x.strip() for x in str(text).split(",")]
    vals = tuple("all" if x == "all" else int(x) for x in parts)
    return vals[0] if len(vals) == 1 else vals


class VectorBestResponse:
    def __init__(self, policy, game, cards="all", chunk=32, seed=0, allin_boards=2000, br_stages=None):
        """``cards``: "all" to enumerate every street's cards, k = cards sampled per public state and
        street (the flop: k flops; turn / river: k cards each, nested), or a tuple per street;
        ``chunk``: boards per batch; ``allin_boards``: sampled boards for pre-flop all-in equities;
        ``br_stages``: the streets (0 = pre-flop, ...) on which the responder deviates - elsewhere it plays the
        policy, so the result splits the exploitability by street (None = all streets, the best response)."""
        self.policy, self.game = policy, game
        self.br_stages = None if br_stages is None else set(br_stages)
        self.dev = policy.device
        self.cards = cards
        self.chunk = chunk
        self.allin_boards = allin_boards
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
        self._allin = {}

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
                        rp = real[p][i]
                        onp[i] = sum(torch.where(rp > 0, real[p][c] / rp.clamp(min=1e-30), 0.0) * onp[c] for c in kids)
                        if self.br_stages is None or nd.stage in self.br_stages:
                            br[i] = torch.stack([br[c] for c in kids]).amax(0)
                        else:  # the responder follows the policy here
                            br[i] = sum(torch.where(rp > 0, real[p][c] / rp.clamp(min=1e-30), 0.0) * br[c] for c in kids)
                    else:
                        br[i] = sum(br[c] for c in kids)
                        onp[i] = sum(onp[c] for c in kids)
            out[p] = (br[0], onp[0])
        return out

    def _opp_mass(self, reach):
        """sum of the opponent's realisation weights over hands without a card of h: (B, 1326)."""
        return reach @ self.disjoint.float()

    def _sign(self, boards):
        """(B, h, h') +1 / -1 / 0 when h beats / loses to / ties h' on the complete ``boards``; 0 for
        hands sharing a card with each other or with the board."""
        s = self._strengths(boards).clamp(max=1 << 20).to(torch.int32)  # blocked hands: INT32_MAX -> 2^20
        ok = s < (1 << 20)
        return torch.sign((s[:, None, :] - s[:, :, None]).float()) * (self.disjoint & ok[:, :, None] & ok[:, None, :])

    def _showdown(self, boards, reach):
        """u[b, h] = sum_h' reach[b, h'] * E[+1 if h beats h', -1 if it loses] over compatible h'; before
        the last street (all-ins) the expectation runs over the rest of the board."""
        if self._sd_boards is not boards:
            self._sd_sign = self._sign(boards) if boards.shape[1] == self.final_cards else self._allin_sign(boards)
            self._sd_boards = boards
        return torch.bmm(self._sd_sign, reach.unsqueeze(-1)).squeeze(-1)

    def _allin_sign(self, boards):
        """Expected showdown sign matrices over the missing board cards: enumerated from the flop on,
        ``allin_boards`` samples pre-flop; masked sums / (N * P(cards miss a hand pair))."""
        n, missing = boards.shape[1], self.final_cards - boards.shape[1]
        out = torch.zeros(len(boards), NUM_COMBOS, NUM_COMBOS, device=self.dev)
        for i, row in enumerate(boards.cpu().numpy()):
            key = tuple(int(c) for c in row)  # only the pre-flop matrix is kept (7 MB per board)
            if key not in self._allin:
                rest = np.setdiff1d(np.arange(NUM_CARDS), row)
                if n >= 3:
                    deals = np.array(list(itertools.combinations(rest, missing)), dtype=np.int64).reshape(-1, missing)
                else:
                    deals = np.stack([np.sort(self.rng.choice(rest, missing, replace=False)) for _ in range(self.allin_boards)])
                full = torch.as_tensor(np.concatenate([np.repeat(row[None], len(deals), 0), deals], axis=1), device=self.dev)
                acc = torch.zeros(NUM_COMBOS, NUM_COMBOS, device=self.dev)
                for c0 in range(0, len(full), 64):
                    acc += self._sign(full[c0 : c0 + 64]).sum(0)
                if n > 0:
                    out[i] = acc / (len(full) * chance_factor(n, self.final_cards))
                    continue
                self._allin[key] = acc / (len(full) * chance_factor(n, self.final_cards))
            out[i] = self._allin[key]
        return out

    def _next_street(self, engine, boards, rho):
        """Values at a street-end leaf (``engine`` = the next street's root): the next street's values
        for every dealt card combination, masked by the hands' compatibility, averaged per parent board."""
        n, n_next = boards.shape[1], BOARD_CARDS_BY_STAGE[int(engine.stage)]
        k = self.cards[BOARD_CARDS_BY_STAGE.index(n)] if isinstance(self.cards, tuple) else self.cards
        kids, parent = self._next_cards(boards, k)
        per_parent = len(kids) // len(boards)
        acc = {p: [torch.zeros(len(boards), NUM_COMBOS, device=self.dev) for _ in range(2)] for p in (0, 1)}
        for c0 in range(0, len(kids), self.chunk):
            kb, kp = kids[c0 : c0 + self.chunk], parent[c0 : c0 + self.chunk]
            ok = self._ok(kb).float()
            vals = self.street(engine_with_board(engine, kb), kb, [x[:, kp] * ok for x in rho])
            for p in (0, 1):
                for j in (0, 1):
                    acc[p][j].index_add_(0, kp, vals[p][j] * ok)
        norm = per_parent * chance_factor(n, n_next)
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
                "br_values": br, "values": onp, "cards": list(self.cards) if isinstance(self.cards, tuple) else self.cards,
                "strategy_queries": self.queries}


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
    from headsup.device import get_device
    from headsup.games.holdem import make_holdem
    from headsup.players import make_player

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="player spec: cfr:..., iterate:..., sdcfr:... (exact average), tab:..., bots")
    p.add_argument("--game", default="fhp", choices=["fhp", "hulh", "nlhe"], help="for players without their own action tree")
    p.add_argument("--cards", default="all", help="'all' (enumerate), k (cards sampled per public state and street) or k1,k2,k3 per street")
    p.add_argument("--allin-boards", type=int, default=2000, help="sampled boards for pre-flop all-in equities")
    p.add_argument("--chunk", type=int, default=32, help="boards per batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--br-streets", default=None,
                   help="streets on which the responder deviates, e.g. 0 (pre-flop) or 1 (flop); default all = the best response")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    device = get_device(args.device)
    player = make_player(args.policy, device=device, seed=args.seed, game=make_holdem(args.game))
    game = getattr(player, "game", None) or make_holdem(args.game)
    cards = parse_cards(args.cards)
    t0 = time.perf_counter()
    res = VectorBestResponse(mixture_policy(player, device, game), game, cards=cards, chunk=args.chunk, seed=args.seed,
                             allin_boards=args.allin_boards,
                             br_stages=None if args.br_streets is None else [int(x) for x in args.br_streets.split(",") if x != ""]).run()
    res["seconds"] = time.perf_counter() - t0
    res["policy"] = args.policy
    print(f"{args.policy}: exploitability {res['exploitability_mbb']:.1f} mbb/g (total {res['total_exploitability_mbb']:.1f}; "
          f"{res['exploitability_chips']:.3f} chips; BR values {res['br_values'][0]:+.3f} / {res['br_values'][1]:+.3f}, "
          f"profile values {res['values'][0]:+.3f} / {res['values'][1]:+.3f}; cards {args.cards}) in {res['seconds']:.0f}s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
