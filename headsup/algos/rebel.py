"""ReBeL on the small poker games: self-play reinforcement learning and search over public belief states.

Brown, Bakhtin, Lerer & Gong 2020 (arXiv 2007.13544), following the official Liar's Dice code
(facebookresearch/rebel) where the paper leaves a choice open and the paper's poker appendix for the chance node:

* a public belief state (PBS) is a public state plus one range (probability vector over the private cards) per
  player; a subgame is one betting round from a PBS - its leaves are the public states where the round is over
  and the board card is due, and their values come from the value network;
* subgames are solved with Linear CFR-D as the official solver does it (alternating updates, the uniform iterate
  inside the average, 1024 steps = 512 updates per player): at EVERY step the network is asked again for the
  leaves' values at the ranges the CURRENT iterate reaches them with; a leaf's value for the traverser is the
  network's output (the value of each hand against the opponent's normalised range) times the opponent's reach;
* the training example of a solved PBS is the linear average over the steps of the iterates' root values, for both
  players; the next PBS is sampled by playing the iterate of a random step t* (one player explores with
  ``epsilon``), ranges updated with that iterate's probabilities; the last round has no leaves and is solved
  without the network; the value of a PBS before the board card is the chance average of the values after it;
* at test time the same search runs with t* drawn from the weights of the average strategy and no exploration:
  the policy played is, in expectation, the solver's average, yet it is consistent with the ranges it hands down
  (playing the average with the average's ranges - "unsafe" search - is exploitable).

The first round's root is always the same PBS and CFR is deterministic, so one root solve per network version
serves every game of a batch; the games' last-round solves run as one vectorised solver (hands x subgames).
Design, the paper-vs-code table and the deviations: docs/superpowers/specs/2026-10-08-rebel-design.md.

    python -m headsup.algos.rebel --game leduc --epochs 300 --seed 0 --json runs/leduc_rebel/leduc_s0.json
    python -m headsup.algos.rebel --game leduc --oracle --iters 1024      # search with exact leaf values, no network
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from headsup.algos.best_response import TabularPolicy, exploitability
from headsup.algos.deep import _load_optimiser

DECISION, FOLD, SHOWDOWN, LEAF = 0, 1, 2, 3
EPS = 1e-80  # the official smoothing: a range without any mass becomes uniform, regret matching never divides by zero
STOPS = ("uniform", "linear")
LEAF_TARGETS = ("net", "solve")
ROOT, BEFORE_BOARD, AFTER_BOARD = 0, 1, 2  # the PBS kinds: start of the first round, end of it, start of the last round


# ----------------------------------------------------------------------------- ranges
def normalise(x, mask=None):
    """Ranges as probability vectors, the official way: 1e-80 is added to every entry (every entry the ``mask``
    allows), so a range the policy leaves no mass at all becomes uniform."""
    x = x + EPS if mask is None else (x + EPS) * mask
    return x / x.sum(-1, keepdims=True)


def bayes_update(belief, probs):
    """The actor's range after a public action: P(hand | action) is proportional to P(hand) P(action | hand)."""
    return normalise(belief * probs)


def board_distribution(beliefs):
    """P(board card | PBS), (..., C), for one-card hands.  The board is uniform over the cards neither player
    holds, so a card's probability is proportional to the mass of the hand pairs (h0 != h1) without it."""
    b0, b1 = beliefs[..., 0, :], beliefs[..., 1, :]
    both = (b0 * b1).sum(-1, keepdims=True)
    w = (b0.sum(-1, keepdims=True) - b0) * (b1.sum(-1, keepdims=True) - b1) - (both - b0 * b1)
    total = w.sum(-1, keepdims=True)
    return np.where(total > 0, w / np.maximum(total, 1e-300), 1.0 / w.shape[-1])


def sample_board(beliefs, rng):
    """One board card per PBS, (M,), drawn from :func:`board_distribution`."""
    cum = np.cumsum(board_distribution(beliefs), axis=-1)
    return np.minimum((rng.random(len(beliefs))[:, None] >= cum).sum(1), beliefs.shape[-1] - 1)


def deal_board(beliefs, board):
    """The public chance step: only the public state changes; the hand the card blocks drops out of both ranges,
    which are renormalised (beliefs (..., 2, H), board (...) card ids)."""
    mask = np.arange(beliefs.shape[-1]) != np.asarray(board)[..., None, None]
    return normalise(beliefs, mask)


def chance_average(values, opp):
    """Values before the board card from the values after it.  ``values[..., c, h]``: hand h's value at the PBS
    with board card c, against the opponent's range renormalised without c; ``opp[..., h']``: the opponent's range
    before the card.  A pair of hands meets each of the other C - 2 cards with equal probability:

        v(h) = 1 / (C - 2) * sum over c != h of mass(c) v_c(h),     mass(c) = the opponent's mass that survives c."""
    C = values.shape[-2]
    mass = opp.sum(-1, keepdims=True) - opp
    return (values * mass[..., :, None] * (1.0 - np.eye(C))).sum(-2) / (C - 2)


def stop_weights(iters, kind):
    """Distribution of the stopping step t* over 0 .. iters: the iterate BEFORE step t* is the one that is played.
    ``uniform``: the official training loop - every step, the odd ones (player 0 one update ahead) and ``iters``
    included.  ``linear``: the official evaluation and what test-time play needs - the even steps below ``iters``
    with weight t* / 2 + 1, the weight the iterate after t* / 2 updates per player has in the average strategy, so
    the policy played is the average in expectation."""
    assert kind in STOPS and iters >= 2 and iters % 2 == 0
    w = np.zeros(iters + 1)
    if kind == "uniform":
        w[:] = 1.0
    else:
        w[0:iters:2] = np.arange(iters // 2) + 1.0
    return w / w.sum()


# ----------------------------------------------------------------------------- the public tree of a betting round
class RoundTree:
    """The public betting tree of one betting round, nodes in depth-first order (a parent before its children).
    A node is a decision of player 0 / 1, a fold, a showdown, or a leaf: the round is over and the board card is
    due - the public states the value network is asked about.  ``amount``: the chips the winner wins at a
    terminal.  ``path[q][n]``: player q's (decision, action) pairs on the way to node n as indices into a strategy
    flattened to (D * A, H), padded with D * A (a row of ones): a player's reach of every node is one gather and
    one product.  Strategies are arrays (..., D, A, H): decision, action, hand."""

    def __init__(self, state):
        A = self.num_actions = state.game.num_actions
        kind, player, child, amount, folder, pot, own = [], [], [], [], [], [], []
        self.hist, self.states, dec = [], {}, []

        def add(s, mine):
            n = len(kind)
            kind.append(DECISION)
            player.append(-1)
            child.append([-1] * A)
            amount.append(0.0)
            folder.append(-1)
            pot.append(float(s.pot))
            own.append(mine)
            self.hist.append(tuple(tuple(h) for h in s.history))
            if s.is_terminal():
                kind[n] = SHOWDOWN if s.folded is None else FOLD
                folder[n] = -1 if s.folded is None else s.folded
                amount[n] = abs(float(s.returns()[0]))
                assert amount[n] > 0, "build the tree from hands that do not tie"
            elif s.is_chance():
                kind[n] = LEAF
                self.states[n] = s
            else:
                p, d = s.current_player, len(dec)
                player[n] = p
                dec.append(n)
                for a in s.legal_actions():
                    step = [list(mine[0]), list(mine[1])]
                    step[p].append(d * A + a)
                    child[n][a] = add(s.child(a), step)
            return n

        add(state, [[], []])
        self.kind, self.player = np.array(kind), np.array(player)
        self.child, self.amount, self.folder, self.pot = np.array(child), np.array(amount), np.array(folder), np.array(pot)
        self.legal = self.child >= 0
        self.num_nodes, self.num_decisions = len(kind), len(dec)
        self.index = {h: n for n, h in enumerate(self.hist)}  # betting history -> node
        self.dec = np.array(dec)
        self.dec_index = np.full(self.num_nodes, -1)
        self.dec_index[self.dec] = np.arange(len(dec))
        self.mine = [np.flatnonzero(self.player[self.dec] == p) for p in (0, 1)]  # each player's decisions
        self.folds, self.shows, self.leaves = (np.flatnonzero(self.kind == k) for k in (FOLD, SHOWDOWN, LEAF))
        self.leaf_index = np.full(self.num_nodes, -1)
        self.leaf_index[self.leaves] = np.arange(len(self.leaves))
        # the fold terminals' sign for each player: the folder loses its chips
        self.fold_sign = np.stack([np.where(self.folder[self.folds] == p, -1.0, 1.0) for p in (0, 1)])
        self.legal_f = self.legal[self.dec].astype(np.float64)[:, :, None]  # (D, A, 1)
        self.uniform = self.legal_f / self.legal_f.sum(1, keepdims=True)
        width = max(len(m[q]) for m in own for q in (0, 1))
        self.path = np.full((2, self.num_nodes, max(width, 1)), self.num_decisions * A)
        for n, m in enumerate(own):
            for q in (0, 1):
                self.path[q, n, : len(m[q])] = m[q]
        # the solver's gathers: the paths to the terminals and leaves (folds, showdowns, leaves - in this order) and
        # to each player's own decisions; children with the node count for illegal actions (a row of zeros)
        self.path_end = self.path[:, np.concatenate([self.folds, self.shows, self.leaves])]
        self.path_own = [self.path[p, self.dec[self.mine[p]]] for p in (0, 1)]
        self.child_pad = np.where(self.legal, self.child, self.num_nodes)

    def reach(self, sigma, q):
        """Player q's own reach of every node under ``sigma`` (B, D, A, H): the product of its action
        probabilities on the way, (B, N, H)."""
        B, D, A, H = sigma.shape
        flat = np.concatenate([sigma.reshape(B, D * A, H), np.ones((B, 1, H))], axis=1)
        return flat[:, self.path[q]].prod(axis=2)

    def numerators(self, sigma, entry=None):
        """``sigma`` (B, D, A, H) weighted by the acting player's reach of each decision (``entry`` (2, B, H): its
        reach of the root).  Summed over strategies - with weights - and normalised per hand, this is the policy
        that plays one of them for the whole game (the paper's definition of an average of policies)."""
        own = np.empty(sigma.shape[:2] + sigma.shape[3:])
        for q in (0, 1):
            x = self.reach(sigma, q)[:, self.dec[self.mine[q]]]
            own[:, self.mine[q]] = x if entry is None else x * entry[q][:, None, :]
        return own[:, :, None, :] * sigma


# ----------------------------------------------------------------------------- the subgame solver
class CFRD:
    """Linear CFR-D on B subgames of one round tree at once, vectorised over subgames and hands - the official
    Liar's Dice solver (subgame_solving.cc, ``CFR`` with ``linear_update``):

    * ``step(p)`` updates ONE player.  It evaluates the current iterate ``last`` (uniform at first): both players'
      reaches from the root ranges, the terminals' exact values against the opponent's reach, and the leaves'
      values ``leaf_fn(p, beliefs) * (the opponent's total reach of the leaf)``, where ``beliefs`` (B, leaves, 2, H)
      are the CURRENT iterate's normalised reaches and ``leaf_fn`` returns each hand's value against the
      opponent's normalised range, (B, leaves, H) - the value network, or an exact solve;
    * the k-th update of a player (k = 0, 1, ...) has weight k + 1: ``root_mean`` is the running mean of the
      iterates' root values with step 2 / (k + 2), regrets and strategy sums are multiplied by (k + 1) / (k + 2)
      after the update; the uniform iterate is inside the average with weight 1;
    * regret matching floors the regrets at ``floor`` = 1e-80: uniform when nothing is positive, and an action
      without positive regret keeps a probability of the order 1e-80 (``floor = 0``: textbook regret matching).

    ``show`` (H, H) or (B, H, H): the sign of a hand against another at a showdown, 0 for ties and impossible
    pairs.  ``mask`` (B, H): 0 for the hand the board blocks.  ``record``: keep every iterate in ``history``.
    Inside, the subgame axis is the last one (every operation of a step then runs over whole blocks of memory);
    the attributes and methods hand out and take arrays with the subgame axis first."""

    def __init__(self, tree, beliefs, show, amount=None, mask=None, leaf_fn=None, record=False, floor=EPS):
        t = self.tree = tree
        self.leaf_fn, self.floor = leaf_fn, floor
        self.beliefs = np.asarray(beliefs, dtype=np.float64)
        B, _, H = self.beliefs.shape
        D, A, N = t.num_decisions, t.num_actions, t.num_nodes
        self.mask = mask
        self._b = np.ascontiguousarray(self.beliefs.transpose(1, 2, 0))  # (2, H, B)
        self._amount = np.ascontiguousarray(np.broadcast_to(t.amount if amount is None else amount, (B, N)).T)[:, None, :]
        self._show = show if show.ndim == 2 else np.ascontiguousarray(show.transpose(1, 2, 0))
        self._mask = None if mask is None else np.ascontiguousarray(mask.T)
        self._flat = np.ones((D * A + 1, H, B))  # the iterate, and a row of ones for the reach gathers
        self._last = self._flat[:-1].reshape(D, A, H, B)
        self._last[:] = t.uniform[..., None]
        self._regret = np.zeros((D, A, H, B))
        self._sum = np.zeros((D, A, H, B))
        for p in (0, 1):  # the uniform strategy is inside the average with weight 1
            self._sum[t.mine[p]] = self._reach(self._flat, t.path_own[p], p)[:, None] * self._last[t.mine[p]]
        self._values_buffer = np.zeros((N + 1, H, B))  # the last row stays zero: the child of an illegal action
        self._root_value, self._root_mean = np.zeros((2, H, B)), np.zeros((2, H, B))
        self.updates, self.steps = [0, 0], 0
        self.history = [self.last.copy()] if record else None

    # the state with the subgame axis first: the iterate, regrets and strategy sums (B, D, A, H), root values (2, B, H)
    last = property(lambda self: np.moveaxis(self._last, -1, 0))
    regret = property(lambda self: np.moveaxis(self._regret, -1, 0))
    strat_sum = property(lambda self: np.moveaxis(self._sum, -1, 0))
    root_value = property(lambda self: self._root_value.transpose(0, 2, 1))
    root_mean = property(lambda self: self._root_mean.transpose(0, 2, 1))

    def reach(self, sigma, q):
        """Player q's reach of every node from the root ranges under ``sigma`` (B, D, A, H): (B, N, H)."""
        return self.beliefs[:, q, None, :] * self.tree.reach(sigma, q)

    def leaf_beliefs(self, sigma):
        """The PBSs ``sigma`` leads to at the leaves: both players' normalised reaches, (B, leaves, 2, H)."""
        mask = None if self.mask is None else self.mask[:, None, :]
        return np.stack([normalise(self.reach(sigma, q)[:, self.tree.leaves], mask) for q in (0, 1)], axis=2)

    def _reach(self, flat, path, q):
        """Player q's reach of the nodes with the paths ``path`` (n, P) under the strategy ``flat``: (n, H, B)."""
        out = flat[path[:, 0]]
        for k in range(1, path.shape[1]):
            out *= flat[path[:, k]]
        out *= self._b[q]
        return out

    def _values(self, p, flat, regret=None):
        """Player p's counterfactual values at every node, (N + 1, H, B) (a buffer: overwritten by the next call),
        when both play the strategy ``flat`` (D * A + 1, H, B); the instantaneous regrets are added to ``regret``."""
        t, V = self.tree, self._values_buffer
        nF, nS = len(t.folds), len(t.shows)
        opp = self._reach(flat, t.path_end[1 - p], 1 - p)  # the opponent's reach of the terminals and leaves
        total = opp.sum(1, keepdims=True)
        if nF:  # every hand of the opponent but the player's own card
            v = (t.fold_sign[p][:, None, None] * self._amount[t.folds]) * (total[:nF] - opp[:nF])
            V[t.folds] = v if self._mask is None else v * self._mask
        if nS:
            V[t.shows] = self._amount[t.shows] * np.einsum("hk...,nk...->nh...", self._show, opp[nF:nF + nS])
        if len(t.leaves):
            own = self._reach(flat, t.path_end[p][nF + nS:], p)
            mask = None if self.mask is None else self.mask[:, None, :]
            pair = (own, opp[nF + nS:]) if p == 0 else (opp[nF + nS:], own)
            beliefs = np.stack([normalise(np.moveaxis(x, -1, 0), mask) for x in pair], axis=2)  # the CURRENT iterate's PBSs
            V[t.leaves] = np.moveaxis(self.leaf_fn(p, beliefs), 0, -1) * total[nF + nS:]
        sigma = flat[:-1].reshape(self._last.shape)
        for d in range(t.num_decisions - 1, -1, -1):
            n = t.dec[d]
            kids = V[t.child_pad[n]]
            if t.player[n] == p:
                v = (sigma[d] * kids).sum(0)
                if regret is not None:
                    regret[d] += (kids - v) * t.legal_f[d][..., None]
            else:  # the opponent's probabilities are inside its reach
                v = kids.sum(0)
            V[n] = v
        return V

    def values(self, p, sigma):
        """Player p's counterfactual values at every node when both play ``sigma`` (B, D, A, H): (B, N, H)."""
        B, D, A, H = sigma.shape
        flat = np.concatenate([np.moveaxis(sigma, 0, -1).reshape(D * A, H, B), np.ones((1, H, B))])
        return np.moveaxis(self._values(p, flat)[:-1], -1, 0).copy()

    def step(self, p=None):
        """One update of player p (default: alternating, player 0 first)."""
        t = self.tree
        p = self.steps % 2 if p is None else p
        root = self._values(p, self._flat, self._regret)[0]
        k = self.updates[p]
        self._root_value[p] = root
        self._root_mean[p] += (root - self._root_mean[p]) * (2.0 / (k + 2))
        mine = t.mine[p]
        pos = np.maximum(self._regret[mine], self.floor)
        pos *= t.legal_f[mine][..., None]
        total = pos.sum(1, keepdims=True)
        if self.floor > 0:  # some legal action always has mass
            pos /= total
        else:
            pos = np.where(total > 0, pos / np.maximum(total, 1e-300), t.uniform[mine][..., None])
        self._last[mine] = pos
        keep = (k + 1.0) / (k + 2.0)
        own = self._reach(self._flat, t.path_own[p], p)
        for i, d in enumerate(mine):
            self._regret[d] *= keep
            self._sum[d] *= keep
            self._sum[d] += own[i] * pos[i]
        self.updates[p] += 1
        self.steps += 1
        if self.history is not None:
            self.history.append(self.last.copy())

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self

    def average(self):
        """The average strategy (B, D, A, H) (uniform for hands that never reach a decision)."""
        total = self._sum.sum(1, keepdims=True)
        avg = np.where(total > 0, self._sum / np.maximum(total, 1e-300), self.tree.uniform[..., None])
        return np.ascontiguousarray(np.moveaxis(avg, -1, 0))


def sample_leaf(tree, sigma, beliefs, epsilon, rng):
    """One walk per game from the root of a solved subgame to a leaf or a terminal (the official
    ``sample_state_to_leaf``).  One player per game - the explorer - takes a uniformly random legal action with
    probability ``epsilon`` at each of its decisions; every other action is drawn from the iterate ``sigma``
    (G, D, A, H) for a hand drawn from the actor's range (given the opponent's: hands exclude each other).
    After EVERY action, explored or not, the actor's range is updated with the iterate's probabilities.
    Returns the nodes reached (G,), the ranges there (G, 2, H) and the explorers (G,)."""
    G = len(sigma)
    node = np.zeros(G, dtype=np.int64)
    beliefs = np.array(beliefs, dtype=np.float64)
    explorer = rng.integers(0, 2, G)
    while True:
        live = np.flatnonzero(tree.kind[node] == DECISION)
        if len(live) == 0:
            return node, beliefs, explorer
        n = node[live]
        j = tree.player[n]
        probs = sigma[live, tree.dec_index[n]]
        own, opp = beliefs[live, j], beliefs[live, 1 - j]
        w = own * (opp.sum(-1, keepdims=True) - opp)  # the actor's hand, given that the opponent holds another card
        w = np.where(w.sum(-1, keepdims=True) > 0, w, own)
        pa = (probs * w[:, None, :]).sum(-1)
        legal = tree.legal[n]
        explore = (j == explorer[live]) & (rng.random(len(live)) < epsilon)
        pa = np.where(explore[:, None], legal / legal.sum(1, keepdims=True), pa)
        cum = np.cumsum(pa, axis=1)
        a = (rng.random(len(live))[:, None] * cum[:, -1:] >= cum).sum(1)  # inverse CDF: never an action of probability 0
        beliefs[live, j] = bayes_update(own, probs[np.arange(len(live)), a])
        node[live] = tree.child[n, a]


# ----------------------------------------------------------------------------- the value network
def value_net(in_dim, out_dim, hidden=256, layers=2):
    """The official network (cfvpy/models.py ``Net2``): Linear -> LayerNorm -> GELU blocks and a linear output
    layer scaled by 0.01, so the first predictions are close to zero."""
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    head = nn.Linear(d, out_dim)
    with torch.no_grad():
        head.weight *= 0.01
        head.bias *= 0.01
    return nn.Sequential(*mods, head)


def huber(x):
    """The official "Huber" loss per output (cfvpy/selfplay.py): x^2 up to |x| = 1, then 2 |x| - 1 - twice the
    textbook form."""
    a = x.abs()
    return torch.where(a <= 1, x.pow(2), 2 * a - 1)


class Replay:
    """Circular buffer of training examples (network input, values of the indexed agent's hands), sampled uniformly."""

    def __init__(self, capacity, x_dim, y_dim):
        self.capacity, self.pos, self.size = int(capacity), 0, 0
        self.x = np.zeros((self.capacity, x_dim), np.float32)
        self.y = np.zeros((self.capacity, y_dim), np.float32)

    def add(self, x, y):
        n = len(x)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.x[idx], self.y[idx] = x, y
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)


# ----------------------------------------------------------------------------- the algorithm
class ReBeL:
    """ReBeL for the Leduc family (one private card per player, at most two betting rounds with one board card
    between them; Kuhn: one round - its only subgame is the whole game and no network is needed).  Search,
    exploration, network, loss, batch and buffer default to the official Liar's Dice configuration; the minibatches
    per epoch and the learning-rate schedule were chosen by measurement on Leduc (the spec has the comparison;
    official: ``steps=50, lr=3e-4, lr_halve_every=400`` over 1000 epochs)."""

    def __init__(self, game, iters=1024, epsilon=0.25, games=2048, steps=200, batch=512, lr=1e-3, lr_halve_every=100, lr_halvings=2,
                 grad_clip=5.0, capacity=2_000_000, hidden=256, layers=2, train_stop="uniform", leaf_targets="net", chance_prob=1.0,
                 eval_iters=None, eval_samples=1024, probe_iters=64, regret_floor=EPS, device="cpu", seed=0):
        assert train_stop in STOPS and leaf_targets in LEAF_TARGETS
        self.game, self.iters, self.epsilon, self.games = game, iters, epsilon, games
        self.steps, self.batch, self.lr, self.lr_halve_every, self.lr_halvings = steps, batch, lr, lr_halve_every, lr_halvings
        self.grad_clip, self.train_stop, self.leaf_targets, self.chance_prob = grad_clip, train_stop, leaf_targets, chance_prob
        self.eval_iters, self.eval_samples, self.probe_iters = eval_iters or iters, eval_samples, min(probe_iters, iters)
        for name, value in (("iters", iters), ("eval_iters", self.eval_iters), ("probe_iters", self.probe_iters)):
            if value < 2 or value % 2:  # here, not at the first evaluation after an epoch of training
                raise ValueError(f"{name} = {value}: a search length counts single-player updates as the official code does "
                                 "and must be a positive even number")
        self.regret_floor, self.device, self.seed = regret_floor, torch.device(device), seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        H = self.num_hands = game.deck_size  # a hand is one card: card c blocks hand c
        # the round trees, from hands and a board card that cannot tie (the builder reads the stakes off the returns)
        root = game.new_initial_state()
        root.apply(0)
        root.apply(1)
        self.tree0 = RoundTree(root)
        self.trees1 = [RoundTree(self.tree0.states[n].child(2)) for n in self.tree0.leaves]
        assert all(len(t.leaves) == 0 and np.array_equal(t.child, self.trees1[0].child) for t in self.trees1), \
            "at most two betting rounds; the last round's tree must not depend on the first round's betting"
        self.tree1 = self.trees1[0] if self.trees1 else None
        self.amount1 = np.stack([t.amount for t in self.trees1]) if self.trees1 else np.zeros((0, 0))  # (leaf, node): the stakes
        # the most a player can win (half the largest pot): the network's values are learned in [-1, 1]
        self.scale = float(max([self.tree0.amount.max()] + [t.amount.max() for t in self.trees1]))
        value = np.array([[game.hand_value(h, b) for h in range(H)] for b in [*range(H), None]], dtype=np.float64)
        self.show = np.sign(value[:, :, None] - value[:, None, :]) * (1.0 - np.eye(H))  # [board (last: none), hand, hand]
        self.hand_mask = 1.0 - np.eye(H)  # [board, hand]
        self.show[:H] *= self.hand_mask[:, :, None] * self.hand_mask[:, None, :]
        self.root_beliefs = np.full((1, 2, H), 1.0 / H)
        self.leaf_pub = self.public(self.tree0.pot[self.tree0.leaves], prechance=True)
        self.in_dim = 3 + H + 2 * H
        self.hidden, self.layers = hidden, layers
        self.net = value_net(self.in_dim, H, hidden, layers).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = Replay(capacity, self.in_dim, H)
        self.epoch = self.examples = self.sgd_steps = self.games_played = 0
        self.last_loss = None
        self._probe = None

    # -- the network's view of a PBS -------------------------------------------------------------------
    def public(self, pot, board=None, prechance=False):
        """Public-state features of M PBSs: [the round is over and the board card is due, pot / the largest
        pot, the board card one-hot (zeros before it)].  The flag separates the end of the first round from its
        start, which have the same pot and no board; the player to act is always player 0 and is left out."""
        pot = np.atleast_1d(np.asarray(pot, dtype=np.float64))
        x = np.zeros((len(pot), 2 + self.num_hands))
        x[:, 0] = float(prechance)
        x[:, 1] = pot / (2.0 * self.scale)
        if board is not None:
            x[np.arange(len(pot)), 2 + np.asarray(board)] = 1.0
        return x

    def encode(self, pub, beliefs, agent):
        """Network input (M, in_dim): [agent index, public features, player 0's range, player 1's range]."""
        M = len(pub)
        return np.concatenate([np.full((M, 1), float(agent)), pub, beliefs.reshape(M, 2 * self.num_hands)], axis=1).astype(np.float32)

    def net_values(self, pub, beliefs, agent):
        """The network's values of the agent's hands at M PBSs, in chips, (M, H): each hand's expected payoff
        against the opponent's normalised range (0 mass for the pairs that share a card)."""
        x = torch.as_tensor(self.encode(pub, beliefs, agent), device=self.device)
        with torch.no_grad():
            out = self.net(x)
        return out.double().cpu().numpy() * self.scale

    def leaf_values(self, p, beliefs):
        """``leaf_fn`` of the first round's solver: the network at the leaves' PBSs, beliefs (B, leaves, 2, H)."""
        B, L = beliefs.shape[:2]
        return self.net_values(np.tile(self.leaf_pub, (B, 1)), beliefs.reshape(B * L, 2, -1), p).reshape(B, L, -1)

    # -- subgames ----------------------------------------------------------------------------------------
    def solver(self, beliefs, leaf=None, board=None, leaf_fn=None, record=False):
        """The solver of B subgames: the first round from the ranges ``beliefs`` (B, 2, H) (``leaf_fn`` values its
        leaves), or - with ``leaf`` (B,) the first round's leaf and ``board`` (B,) the card - the last round,
        which ends in terminals only: no leaf values, no network."""
        if board is None:
            return CFRD(self.tree0, beliefs, self.show[-1], leaf_fn=leaf_fn, record=record, floor=self.regret_floor)
        return CFRD(self.tree1, beliefs, self.show[board], amount=self.amount1[leaf], mask=self.hand_mask[board], record=record,
                    floor=self.regret_floor)

    def search(self, iters=None, leaf_fn=None, record=True):
        """The root subgame solved for ``iters`` steps with the network at the leaves (or ``leaf_fn``)."""
        leaf_fn = (leaf_fn or self.leaf_values) if len(self.tree0.leaves) else None
        return self.solver(self.root_beliefs, leaf_fn=leaf_fn, record=record).run(self.iters if iters is None else iters)

    def before_board(self, values, beliefs):
        """Both players' values (M, 2, H) at M PBSs before the board card from their values after each card,
        ``values`` (2, M, C, H)."""
        return np.stack([chance_average(values[p], beliefs[:, 1 - p]) for p in (0, 1)], axis=1)

    def solve_boards(self, leaf, beliefs, iters):
        """M PBSs at the end of the first round (leaf index, ranges) solved to the end of the game: the last
        round for every board card, ``iters`` steps.  Returns their values (M, 2, H) and the solver (rows: PBS x card)."""
        M, C = len(leaf), self.num_hands
        board = np.tile(np.arange(C), M)
        sub = self.solver(deal_board(np.repeat(beliefs, C, axis=0), board), np.repeat(leaf, C), board).run(iters)
        return self.before_board(sub.root_mean.reshape(2, M, C, self.num_hands), beliefs), sub

    def exact_leaf(self, iters):
        """A ``leaf_fn`` that solves the rest of the game instead of asking the network (what a perfect value
        network of a CFR-D search with ``iters`` steps would return)."""
        def leaf_fn(p, beliefs):
            B, L = beliefs.shape[:2]
            values, _ = self.solve_boards(np.tile(np.arange(L), B), beliefs.reshape(B * L, 2, -1), iters)
            return values[:, p].reshape(B, L, -1)

        return leaf_fn

    # -- self-play ----------------------------------------------------------------------------------------
    def self_play(self, search, games=None):
        """``games`` playthroughs below the solved root: per game a stopping step, a walk of that iterate to a
        leaf (one player explores), the board card, and the last round's solve.  Returns the PBSs met with their
        targets - ``stage``, ``leaf``, ``board`` (-1: none), ``beliefs`` (M, 2, H), ``values`` (M, 2, H) in chips -
        and per game ``stop``, ``node``, ``explorer``.  Targets: a solved PBS gets the solve's ``root_mean``; a PBS
        before the board card gets the chance average of the values after each card - the network's own
        (``leaf_targets = net``, the paper's poker agent) or exact solves' (``solve``)."""
        G, rng, t0, T = games or self.games, self.rng, self.tree0, search.steps
        stop = rng.choice(T + 1, size=G, p=stop_weights(T, self.train_stop))
        iterates = np.concatenate(search.history)
        node, beliefs, explorer = sample_leaf(t0, iterates[stop], np.repeat(self.root_beliefs, G, axis=0), self.epsilon, rng)
        out = {"stop": stop, "node": node, "explorer": explorer}
        stage, leaf, board = [np.array([ROOT])], [np.array([-1])], [np.array([-1])]
        ranges, values = [self.root_beliefs], [search.root_mean.transpose(1, 0, 2)]
        at = np.flatnonzero(t0.kind[node] == LEAF)
        if len(at):
            lf, lb = t0.leaf_index[node[at]], beliefs[at]
            pre = np.flatnonzero(rng.random(len(at)) < self.chance_prob)
            if self.leaf_targets == "net":  # the game goes on: the card, and the last round's solve from the PBS after it
                card = sample_board(lb, rng)
                post = deal_board(lb, card)
                sub = self.solver(post, lf, card).run(T)
                stage.append(np.full(len(at), AFTER_BOARD))
                leaf.append(lf)
                board.append(card)
                ranges.append(post)
                values.append(sub.root_mean.transpose(1, 0, 2))
            if len(pre):  # exact targets (every card's subgame solved) or the paper's (the network after every card)
                solved = self.leaf_targets == "solve"
                stage.append(np.full(len(pre), BEFORE_BOARD))
                leaf.append(lf[pre])
                board.append(np.full(len(pre), -1))
                ranges.append(lb[pre])
                values.append(self.solve_boards(lf[pre], lb[pre], T)[0] if solved else self.net_before_board(lf[pre], lb[pre]))
        out.update(stage=np.concatenate(stage), leaf=np.concatenate(leaf), board=np.concatenate(board),
                   beliefs=np.concatenate(ranges), values=np.concatenate(values))
        return out

    def net_before_board(self, leaf, beliefs):
        """Values (M, 2, H) of PBSs at the end of the first round as the paper's poker agent makes these targets:
        the network is asked for the values after every board card and they are averaged with card removal."""
        M, C = len(leaf), self.num_hands
        board = np.tile(np.arange(C), M)
        pub = self.public(np.repeat(self.tree0.pot[self.tree0.leaves][leaf], C), board)
        post = deal_board(np.repeat(beliefs, C, axis=0), board)
        after = np.stack([self.net_values(pub, post, p) for p in (0, 1)])
        return self.before_board(after.reshape(2, M, C, -1), beliefs)

    def examples_of(self, out):
        """Network inputs and targets of the PBSs of :meth:`self_play`: one row per PBS and agent."""
        stage, leaf, board = out["stage"], out["leaf"], out["board"]
        pot = np.full(len(stage), self.tree0.pot[0])
        pot[stage != ROOT] = self.tree0.pot[self.tree0.leaves][leaf[stage != ROOT]]
        pub = self.public(pot)
        pub[:, 0] = stage == BEFORE_BOARD
        dealt = np.flatnonzero(board >= 0)
        pub[dealt, 2 + board[dealt]] = 1.0
        x = np.concatenate([self.encode(pub, out["beliefs"], agent) for agent in (0, 1)])
        y = np.concatenate([out["values"][:, agent] for agent in (0, 1)]) / self.scale
        return x, y.astype(np.float32)

    def generate(self):
        """One batch of self-play with the current network: the root solve (the same for every game: the root PBS
        is fixed and CFR is deterministic), ``games`` playthroughs, their examples into the buffer."""
        out = self.self_play(self.search())
        x, y = self.examples_of(out)
        self.buffer.add(x, y)
        self.examples += len(x)
        self.games_played += len(out["stop"])
        return out

    # -- training -----------------------------------------------------------------------------------------
    def loss(self, x, y):
        """The official loss: the "Huber" of the error of each hand's value (in units of the largest win), mean
        over hands and examples."""
        return huber(self.net(x) - y).mean()

    def learning_rate(self, epoch):
        """Halved every ``lr_halve_every`` epochs, at most ``lr_halvings`` times."""
        return self.lr * 0.5 ** min(epoch // self.lr_halve_every, self.lr_halvings)

    def train(self, steps):
        buf = self.buffer
        if buf.size == 0 or steps <= 0:
            return
        for group in self.opt.param_groups:
            group["lr"] = self.learning_rate(self.epoch)
        for _ in range(steps):
            idx = self.rng.integers(0, buf.size, self.batch)
            loss = self.loss(torch.as_tensor(buf.x[idx], device=self.device), torch.as_tensor(buf.y[idx], device=self.device))
            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.opt.step()
            self.sgd_steps += 1
        self.last_loss = loss.item()

    def iterate(self, n=1):
        """``n`` epochs: self-play with the current network, then ``steps`` minibatches."""
        for _ in range(n):
            self.generate()
            self.train(self.steps)
            self.epoch += 1
        return self

    # -- play and evaluation ---------------------------------------------------------------------------------
    def table(self, num0, num1=None):
        """info_key -> probabilities from strategy numerators (any positive multiple of the behaviour strategy
        per hand): the first round's (D, A, H) and the last round's (leaf, card, D, A, H)."""
        table = {}

        def put(tree, num, board, rnd):
            total = num.sum(1)
            for d, n in enumerate(tree.dec):
                for h in range(self.num_hands):
                    if h != board and total[d, h] > 0:
                        table[(int(tree.player[n]), h, board, rnd, tree.hist[n])] = num[d, :, h] / total[d, h]

        put(self.tree0, num0, None, 0)
        if num1 is not None:
            for leaf, tree in enumerate(self.trees1):
                for b in range(self.num_hands):
                    put(tree, num1[leaf, b], b, 1)
        return table

    def playthrough(self, rng, search=None, stop0=None, stop1=None):
        return Playthrough(self, search or self.search(self.eval_iters), rng, stop0, stop1)

    def policies(self, search, samples=0, rng=None, value_error=False):
        """The policies behind a root solve of T steps, as tables for the exact best response:

        * ``exact``: what ReBeL plays in expectation.  A playthrough stops the root solve at an even step 2 j with
          probability proportional to j + 1, plays that iterate, hands its ranges to the last round's solve and
          stops that one the same way; the mixture over the last round's stopping steps is that solve's average
          strategy after T - 2 steps, so the policy is the mixture over j of [iterate j, the last round's
          averages at iterate j's ranges] - all T / 2 x leaves x cards solves in one batch;
        * ``sampled``: the paper's evaluation - the average of ``samples`` playthrough policies (``stop0`` (K,) and
          ``stop1`` (K, leaf, card): their stopping steps);
        * ``unsafe``: the root's average strategy with the last round solved at the average's ranges.

        ``value_error``: the RMS error in chips of the leaf values this search was fed - the network's output times
        the opponent's reach, at the leaf PBSs of the root iterates (weights j + 1) - against the exact solves."""
        t0, T, H = self.tree0, search.steps, self.num_hands
        J = T // 2
        w = stop_weights(T, "linear")[0:T:2]
        iterates = np.concatenate(search.history)[0:T:2]  # after j updates per player
        num0 = t0.numerators(iterates)
        out = {"value_error": None}
        if samples:
            out["stop0"] = 2 * rng.choice(J, size=samples, p=w)
        average = search.average()
        if not len(t0.leaves):
            out["exact"] = TabularPolicy(self.game, self.table(np.tensordot(w, num0, 1)))
            out["unsafe"] = TabularPolicy(self.game, self.table(t0.numerators(average)[0]))
            if samples:
                out["sampled"] = TabularPolicy(self.game, self.table(num0[out["stop0"] // 2].sum(0)))
            return out
        t1, L, C = self.tree1, len(t0.leaves), H
        # rows (j, leaf, card): the last round at the ranges iterate j reaches the leaf with
        beliefs = search.leaf_beliefs(iterates)
        leaf, board = np.tile(np.repeat(np.arange(L), C), J), np.tile(np.arange(C), J * L)
        sub = self.solver(deal_board(np.repeat(beliefs.reshape(J * L, 2, H), C, axis=0), board), leaf, board)
        entry = np.stack([np.repeat(t0.reach(iterates, q)[:, t0.leaves].reshape(J * L, H), C, axis=0) for q in (0, 1)])
        shape = (L, C, t1.num_decisions, t1.num_actions, H)
        if samples:  # the snapshots: playthrough k stops the solve below (leaf, card) at step stop1[k, leaf, card]
            out["stop1"] = 2 * rng.choice(J, size=(samples, L, C), p=w)
            rows = ((out["stop0"] // 2)[:, None, None] * L + np.arange(L)[None, :, None]) * C + np.arange(C)[None, None, :]
            order = np.argsort(out["stop1"].ravel(), kind="stable")
            rows, due = rows.ravel()[order], out["stop1"].ravel()[order] // 2
            first = np.searchsorted(due, np.arange(J + 1))
            sampled = np.zeros((L * C,) + shape[2:])
        for j in range(J):
            if samples and first[j + 1] > first[j]:
                r = rows[first[j]:first[j + 1]]
                np.add.at(sampled, r % (L * C), t1.numerators(sub.last[r], entry[:, r]))
            if j < J - 1:
                sub.run(2)
        num1 = t1.numerators(sub.average(), entry).reshape((J,) + shape)
        out["exact"] = TabularPolicy(self.game, self.table(np.tensordot(w, num0, 1), np.tensordot(w, num1, 1)))
        if samples:
            out["sampled"] = TabularPolicy(self.game, self.table(num0[out["stop0"] // 2].sum(0), sampled.reshape(shape)))
        if value_error:
            exact = self.before_board(sub.root_mean.reshape(2, J * L, C, H), beliefs.reshape(J * L, 2, H))
            net = np.stack([self.leaf_values(p, beliefs) for p in (0, 1)], axis=2).reshape(J * L, 2, H)
            fed = np.stack([search.reach(iterates, 1 - p)[:, t0.leaves].sum(-1) for p in (0, 1)], axis=2)  # the opponent's reach
            error = (net - exact).reshape(J, L, 2, H) * fed[..., None]
            out["value_error"] = float(np.sqrt(w @ (error ** 2).reshape(J, -1).mean(1)))
        # unsafe search: the average strategy, and the last round at the ranges of the average
        lb = search.leaf_beliefs(average)[0]
        leaf, board = np.repeat(np.arange(L), C), np.tile(np.arange(C), L)
        last = self.solver(deal_board(np.repeat(lb, C, axis=0), board), leaf, board).run(T)
        entry = np.stack([np.repeat(t0.reach(average, q)[0, t0.leaves], C, axis=0) for q in (0, 1)])
        out["unsafe"] = TabularPolicy(self.game, self.table(t0.numerators(average)[0], t1.numerators(last.average(), entry).reshape(shape)))
        return out

    def probe(self):
        """A fixed set of PBSs at the end of the first round with exact values, to follow the network's error:
        the leaf PBSs of the even iterates of a short search with exact leaf values (weights as at test time),
        each solved to the end of the game for ``iters`` steps.  None when the game has one round."""
        if self._probe is None and len(self.tree0.leaves):
            T, L = self.probe_iters, len(self.tree0.leaves)
            search = self.search(T, leaf_fn=self.exact_leaf(T))
            beliefs = search.leaf_beliefs(np.concatenate(search.history)[0:T:2])
            values, _ = self.solve_boards(np.tile(np.arange(L), T // 2), beliefs.reshape(-1, 2, self.num_hands), self.iters)
            self._probe = (beliefs, values, stop_weights(T, "linear")[0:T:2])
        return self._probe

    def value_error(self):
        """RMS error of the network in chips on the probe set (None for a one-round game)."""
        if self.probe() is None:
            return None
        beliefs, values, w = self._probe
        net = np.stack([self.leaf_values(p, beliefs) for p in (0, 1)], axis=2).reshape(values.shape)
        return float(np.sqrt(w @ ((net - values) ** 2).reshape(len(w), -1).mean(1)))

    def root_value(self, search):
        """Player 0's value of the game according to a root solve: its average root values over the hands."""
        b = self.root_beliefs[0]
        return float((b[0] * search.root_mean[0, 0]).sum() / (1.0 - (b[0] * b[1]).sum()))

    def evaluate(self):
        """Exact exploitability (chips = antes, mean over the two seats) of the test-time policy: ``exploitability``
        - of the policy played in expectation; ``exploitability_sampled`` - of the average of ``samples`` sampled
        playthrough policies, the paper's protocol (in expectation an upper bound of the former: the best
        response's value is convex in the policy); ``exploitability_unsafe`` - of unsafe search.  The stopping
        steps come from a generator of their own: a run does not depend on how often it is evaluated."""
        search = self.search(self.eval_iters)
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        out = self.policies(search, self.eval_samples, rng, value_error=True)
        return {"exploitability": float(exploitability(self.game, out["exact"])[0]),
                "exploitability_sampled": float(exploitability(self.game, out["sampled"])[0]) if self.eval_samples else None,
                "samples": self.eval_samples,
                "exploitability_unsafe": float(exploitability(self.game, out["unsafe"])[0]),
                "root_value": self.root_value(search), "value_error": self.value_error(), "value_error_search": out["value_error"]}

    # -- checkpoints ----------------------------------------------------------------------------------------
    def state_dict(self):
        buf = self.buffer
        return {
            "game": self.game.name, "in_dim": self.in_dim, "hidden": self.hidden, "layers": self.layers, "seed": self.seed,
            "epoch": self.epoch, "examples": self.examples, "sgd_steps": self.sgd_steps, "games_played": self.games_played,
            "last_loss": self.last_loss,
            "net": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
            # a deep copy: an optimiser's state_dict() hands out its live tensors (a snapshot must not keep changing)
            "opt": copy.deepcopy(self.opt.state_dict()),
            "buffer": {"capacity": buf.capacity, "pos": buf.pos, "size": buf.size,
                       "x": buf.x[: buf.size].copy(), "y": buf.y[: buf.size].copy()},
            "rng": self.rng.bit_generator.state,  # all the randomness after the network's initialisation
        }

    def load_state_dict(self, state):
        mine = (self.game.name, self.in_dim, self.hidden, self.layers)
        saved = (state["game"], state["in_dim"], state["hidden"], state["layers"])
        if saved != mine:
            raise ValueError(f"checkpoint of (game, inputs, hidden, layers) = {saved} cannot be loaded into a solver of {mine}")
        if state["buffer"]["capacity"] != self.buffer.capacity:  # a FIFO: its write position means nothing in a buffer of another size
            raise ValueError(f"checkpoint with a replay capacity of {state['buffer']['capacity']} cannot be loaded into one of "
                             f"{self.buffer.capacity}")
        self.epoch, self.examples, self.sgd_steps = int(state["epoch"]), int(state["examples"]), int(state["sgd_steps"])
        self.games_played, self.last_loss, self.seed = int(state["games_played"]), state["last_loss"], state["seed"]
        self.net.load_state_dict(state["net"])
        _load_optimiser(self.opt, state["opt"])
        buf, saved = self.buffer, state["buffer"]
        buf.size, buf.pos = int(saved["size"]), int(saved["pos"])
        buf.x[: buf.size], buf.y[: buf.size] = saved["x"], saved["y"]
        self.rng.bit_generator.state = state["rng"]
        return self


class Playthrough:
    """The policy ReBeL plays in one hand, ``policy(state) -> probabilities``: the root solve stopped at ``stop0``
    (drawn with the average strategy's weights) is played for the whole first round; after the board card the last
    round is solved from the ranges that iterate gives BOTH players at the leaf, stopped at a step drawn the same
    way, and that iterate is played to the end.  One stopping step per subgame, for all its decisions."""

    def __init__(self, solver, search, rng=None, stop0=None, stop1=None):
        self.solver, self.search, self.rng, self.stop1 = solver, search, rng, stop1
        self.weights = stop_weights(search.steps, "linear")
        self.stop0 = int(rng.choice(len(self.weights), p=self.weights)) if stop0 is None else int(stop0)
        self.sigma0 = search.history[self.stop0]
        self.beliefs = search.leaf_beliefs(self.sigma0)[0] if len(solver.tree0.leaves) else None
        self.round1 = {}  # (leaf, card) -> (stopping step, strategy)

    def __call__(self, state):
        s, hist, h = self.solver, tuple(tuple(x) for x in state.history), state.cards[state.current_player]
        t0 = s.tree0
        if state.round == 0:
            return self.sigma0[0, t0.dec_index[t0.index[hist]], :, h]
        leaf, card = int(t0.leaf_index[t0.index[(hist[0], ())]]), state.board
        if (leaf, card) not in self.round1:
            if self.stop1 is None:
                stop = int(self.rng.choice(len(self.weights), p=self.weights))
            else:
                stop = int(self.stop1 if np.ndim(self.stop1) == 0 else self.stop1[leaf, card])
            sub = s.solver(deal_board(self.beliefs[leaf][None], np.array([card])), np.array([leaf]), np.array([card])).run(stop)
            self.round1[(leaf, card)] = (stop, sub.last[0])
        tree = s.trees1[leaf]
        return self.round1[(leaf, card)][1][tree.dec_index[tree.index[hist]], :, h]


def main(argv=None):
    from headsup.games import make_small_game

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--epochs", type=int, default=300, help="one epoch: a root solve, --games playthroughs, --steps minibatches "
                   "(the official Liar's Dice run: --epochs 1000 --steps 50 --lr 3e-4 --lr-halve-every 400)")
    p.add_argument("--iters", type=int, default=1024, help="search steps per subgame, single-player updates (1024 = 512 per player)")
    p.add_argument("--eval-iters", type=int, default=None, help="search steps at evaluation (default: --iters)")
    p.add_argument("--games", type=int, default=2048, help="playthroughs per epoch")
    p.add_argument("--epsilon", type=float, default=0.25, help="the exploring player's probability of a random action")
    p.add_argument("--train-stop", default="uniform", choices=STOPS,
                   help="stopping step in self-play: uniform over all steps (the official code) or the test-time weights "
                   "(the paper's Algorithm 2)")
    p.add_argument("--leaf-targets", default="net", choices=LEAF_TARGETS,
                   help="targets before the board card: the network's values after every card (the paper) or exact solves of every card")
    p.add_argument("--regret-floor", type=float, default=EPS,
                   help="regret matching floors the regrets at this value (the official code: 1e-80; 0: the textbook rule)")
    p.add_argument("--chance-prob", type=float, default=1.0,
                   help="share of the reached end-of-round PBSs that become examples (the paper: 1/3)")
    p.add_argument("--steps", type=int, default=200, help="minibatches per epoch")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-halve-every", type=int, default=100, help="epochs between halvings of the learning rate (at most twice)")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--buffer", type=int, default=2_000_000, help="replay capacity in examples (FIFO)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=20, help="evaluate every this many epochs (and at the first and the last)")
    p.add_argument("--eval-samples", type=int, default=1024, help="playthrough policies in the sampled mixture (the paper: 1024)")
    p.add_argument("--oracle", action="store_true", help="no network: search with exact leaf values at --iters steps and evaluate it")
    p.add_argument("--checkpoint", default=None, help="saved at evaluations; an existing file is resumed from")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0,
                   help="at most one checkpoint per this many minutes (plus the final one)")
    p.add_argument("--device", default="cpu", help="the network's device; the search itself is numpy on the CPU")
    p.add_argument("--threads", type=int, default=1,
                   help="torch's CPU threads. A seed gives the same run only with the same number: it changes the order of the "
                   "network's float additions and the search amplifies the last digit (2 threads: about 10 %% faster per epoch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    game = make_small_game(args.game)
    solver = ReBeL(game, iters=args.iters, epsilon=args.epsilon, games=args.games, steps=args.steps, batch=args.batch, lr=args.lr,
                   lr_halve_every=args.lr_halve_every, grad_clip=args.grad_clip, capacity=args.buffer, hidden=args.hidden,
                   layers=args.layers, train_stop=args.train_stop, leaf_targets=args.leaf_targets, chance_prob=args.chance_prob,
                   eval_iters=args.eval_iters, eval_samples=args.eval_samples, regret_floor=args.regret_floor, device=args.device,
                   seed=args.seed)
    curve, elapsed = [], 0.0

    def dump():
        if args.json:  # atomic: a crash while writing leaves the previous curve intact
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json + ".tmp", "w") as f:
                algo = "rebel-oracle" if args.oracle else "rebel"
                json.dump({"game": args.game, "algo": algo, "args": vars(args), "curve": curve}, f, indent=2)
            os.replace(args.json + ".tmp", args.json)

    if args.oracle:  # the calibration: what the search alone reaches with a perfect value function
        t0 = time.perf_counter()
        search = solver.search(args.iters, leaf_fn=solver.exact_leaf(args.iters))
        out = solver.policies(search, args.eval_samples, np.random.default_rng(args.seed))
        row = {"iters": args.iters, "exploitability": float(exploitability(game, out["exact"])[0]),
               "exploitability_sampled": float(exploitability(game, out["sampled"])[0]) if args.eval_samples else None,
               "samples": args.eval_samples, "exploitability_unsafe": float(exploitability(game, out["unsafe"])[0]),
               "root_value": solver.root_value(search), "seconds": time.perf_counter() - t0}
        curve.append(row)
        print(f"{args.game} rebel, exact leaf values, {args.iters} steps: exploitability {row['exploitability']:.4f}  "
              f"sampled ({args.eval_samples}) {_fmt(row['exploitability_sampled'])}  unsafe {row['exploitability_unsafe']:.4f}  "
              f"root value {row['root_value']:+.4f}  ({row['seconds']:.0f}s)", flush=True)
        dump()
        return
    if args.checkpoint and os.path.exists(args.checkpoint):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        solver.load_state_dict(saved["solver"])
        curve, elapsed = saved["curve"], saved["seconds"]
        print(f"resumed from {args.checkpoint} at epoch {solver.epoch}", flush=True)
    t0 = time.perf_counter() - elapsed
    last_save = time.perf_counter()
    for epoch in range(solver.epoch + 1, args.epochs + 1):
        solver.iterate()
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            ev = solver.evaluate()
            curve.append({"epoch": epoch, "examples": solver.examples, "sgd_steps": solver.sgd_steps, "games": solver.games_played,
                          **ev, "loss": solver.last_loss, "seconds": time.perf_counter() - t0})
            print(f"{args.game} rebel epoch {epoch}: exploitability {ev['exploitability']:.4f}  sampled ({ev['samples']}) "
                  f"{_fmt(ev['exploitability_sampled'])}  unsafe {ev['exploitability_unsafe']:.4f}  root value {ev['root_value']:+.4f}  "
                  f"value error {_fmt(ev['value_error'])} / search {_fmt(ev['value_error_search'])}  examples {solver.examples}  "
                  f"sgd steps {solver.sgd_steps}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            dump()
            due = time.perf_counter() - last_save >= 60.0 * args.checkpoint_minutes or epoch == args.epochs
            if args.checkpoint and due:  # atomic: a crash while writing leaves the previous checkpoint intact
                torch.save({"solver": solver.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}, args.checkpoint + ".tmp")
                os.replace(args.checkpoint + ".tmp", args.checkpoint)
                last_save = time.perf_counter()
    dump()


def _fmt(x):
    return "–" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    main()
