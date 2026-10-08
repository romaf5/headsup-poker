"""Deep (Predictive) Discounted CFR on the game protocol: VR-DeepDCFR+ and VR-DeepPDCFR+.

Xu, Li, Fu, Fu, Xing & Cheng 2025 (arXiv 2511.08174), following the authors' code (rpSebastian/DeepPDCFR):

* per player one persistent network ``R`` of cumulative *advantages*, fitted every iteration to its own previous
  output - clipped at zero and discounted by ``d_t = (t-1)^alpha / ((t-1)^alpha + c)`` - plus the advantages
  sampled in this iteration only (no reservoir, no iteration weights); the clip is applied where ``R`` is read;
* VR-DeepPDCFR+ adds a network ``r`` of the latest instantaneous advantages and plays regret matching on the
  predicted next regret ``max(R, 0) d_t + r``;
* outcome sampling (the traverser explores with ``eps``), variance-reduced by a history-action baseline ``Q``
  (player 0's value; negated for player 1) that is re-fitted from scratch with expected-SARSA targets;
* the result is an average-strategy network fitted on a reservoir of ``(I, t, sigma_t(I))`` with weights
  ``(t / T)^gamma``.

Where the paper and the code disagree the defaults follow the code (see docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md).
Small games only: the whole tree is compiled to arrays and the episodes of an iteration are sampled in one
vectorised pass.

    python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 10000000 --json runs/x.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from headsup.algos.best_response import TabularPolicy, exploitability
from headsup.algos.deep import _adam, _batch_index, _optimise
from headsup.deepcfr.memory import ReservoirBuffer

VARIANTS = ("dcfr+", "pdcfr+")
# the authors' configs: alpha, gamma and the discount denominator's constant (the paper's formula has 1 for both)
DEFAULTS = {"dcfr+": dict(alpha=2.0, gamma=2.0, offset=1.5), "pdcfr+": dict(alpha=2.3, gamma=2.0, offset=1.0)}
DECISION, CHANCE, TERMINAL = 0, 1, 2


def discount(t, alpha, offset):
    """d_t = (t-1)^alpha / ((t-1)^alpha + offset): how much of the clipped cumulative advantage survives into
    iteration t (0 at t = 1)."""
    x = float(t - 1) ** alpha
    return x / (x + offset)


def strategy_rows(R, r, legal, d, variant, fallback="authors"):
    """Current strategies for rows of network outputs: regret matching on ``R`` (dcfr+) or on the predicted
    regret ``max(R, 0) d + r`` (pdcfr+), over the legal actions.  When nothing is positive: ``authors`` - the
    largest raw output (dcfr+) / the first legal action (pdcfr+: their code clips before the argmax);
    ``argmax`` - the largest unclipped score; ``uniform``."""
    R = np.where(legal, R, 0.0)
    score = R if variant == "dcfr+" else np.maximum(R, 0.0) * d + np.where(legal, r, 0.0)
    pos = np.maximum(score, 0.0)
    total = pos.sum(-1, keepdims=True)
    if fallback == "uniform":
        fb = legal / legal.sum(-1, keepdims=True)
    else:
        if fallback == "authors" and variant == "pdcfr+":
            best = np.argmax(legal, axis=-1)
        else:
            best = np.argmax(np.where(legal, score, -np.inf), axis=-1)
        fb = np.zeros_like(pos)
        np.put_along_axis(fb, best[..., None], 1.0, axis=-1)
    return np.where(total > 0, pos / np.maximum(total, 1e-300), fb)


def mlp(in_dim, out_dim, hidden=64, layers=3):
    """The authors' network: ReLU layers initialised from a normal truncated at two standard deviations of
    1 / sqrt(fan_in) with zero bias, and an output layer that starts at zero."""
    mods, d = [], in_dim
    for _ in range(layers):
        lin = nn.Linear(d, hidden)
        sd = 1.0 / math.sqrt(d)
        nn.init.trunc_normal_(lin.weight, std=sd, a=-2 * sd, b=2 * sd)
        nn.init.zeros_(lin.bias)
        mods += [lin, nn.ReLU()]
        d = hidden
    head = nn.Linear(d, out_dim)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*mods, head)


class Tree:
    """The whole game as arrays: nodes in depth-first order (a parent's id is smaller than its children's).
    Decision nodes carry their infoset index (``info``) and their index among the decision nodes (``dec``: a
    row of ``hist_x``, the concatenated infostates of both players - the baseline's input)."""

    def __init__(self, game):
        A = self.num_actions = game.num_actions
        kind, player, info, dec, util, child, legal, hist_x, chance = [], [], [], [], [], [], [], [], {}
        index, self.info_keys, info_obs, info_legal, info_player = {}, [], [], [], []
        self.depth = 0  # the largest number of decisions on a path

        def add(state, depth):
            i = len(kind)
            kind.append(DECISION)
            player.append(-1)
            info.append(-1)
            dec.append(-1)
            util.append(0.0)
            child.append([-1] * A)
            legal.append([False] * A)
            if state.is_terminal():
                kind[i], util[i] = TERMINAL, float(state.returns()[0])
                self.depth = max(self.depth, depth)
            elif state.is_chance():
                kind[i] = CHANCE
                outcomes = state.chance_outcomes()
                chance[i] = ([add(state.child(a), depth) for a, _ in outcomes], [p for _, p in outcomes])
            else:
                p = state.current_player
                key = state.info_key(p)
                if key not in index:
                    index[key] = len(self.info_keys)
                    self.info_keys.append(key)
                    info_obs.append(state.info_state(p))
                    info_legal.append(state.legal_mask())
                    info_player.append(p)
                player[i], info[i], dec[i] = p, index[key], len(hist_x)
                hist_x.append(np.concatenate([state.info_state(0), state.info_state(1)]))
                for a in state.legal_actions():
                    legal[i][a] = True
                    child[i][a] = add(state.child(a), depth + 1)
            return i

        add(game.new_initial_state(), 0)
        self.kind = np.array(kind, dtype=np.int8)
        self.player = np.array(player, dtype=np.int8)
        self.info = np.array(info, dtype=np.int64)
        self.dec = np.array(dec, dtype=np.int64)
        self.util = np.array(util)  # player 0's utility at the terminals
        self.child = np.array(child, dtype=np.int64)
        self.legal = np.array(legal, dtype=bool)
        self.hist_x = np.stack(hist_x).astype(np.float32)
        self.info_obs = np.stack(info_obs).astype(np.float32)
        self.info_legal = np.stack(info_legal).astype(bool)
        self.info_player = np.array(info_player)
        self.num_nodes, self.num_infosets, self.num_decisions = len(kind), len(self.info_keys), len(hist_x)
        width = max(len(kids) for kids, _ in chance.values())
        self.chance_child = np.full((self.num_nodes, width), -1, dtype=np.int64)
        self.chance_prob = np.zeros((self.num_nodes, width))
        self.chance_cum = np.full((self.num_nodes, width), 2.0)  # padding above every uniform draw
        for i, (kids, probs) in chance.items():
            self.chance_child[i, : len(kids)] = kids
            self.chance_prob[i, : len(kids)] = probs
            cum = np.cumsum(probs)
            cum[-1] = 1.0
            self.chance_cum[i, : len(kids)] = cum
        self.max_utility = float(np.abs(self.util).max())

    def values(self, sigma):
        """Player 0's expected utility at every node under the profile ``sigma`` (num_infosets, A)."""
        v = self.util.copy()
        for i in range(self.num_nodes - 1, -1, -1):
            if self.kind[i] == CHANCE:
                k = self.chance_child[i] >= 0
                v[i] = self.chance_prob[i, k] @ v[self.chance_child[i, k]]
            elif self.kind[i] == DECISION:
                a = self.legal[i]
                v[i] = sigma[self.info[i], a] @ v[self.child[i, a]]
        return v

    def advantages(self, sigma, p):
        """Exact advantages of player p under ``sigma``: for each of p's infosets I and legal action a,
        sum_h pi_-p(h) (v_p(h a) - v_p(h)) / sum_h pi_-p(h) over the histories h of I (pi_-p: the opponent's
        and chance's reach).  (num_infosets, A); zero for the other player's and for unreachable infosets."""
        v = self.values(sigma) * (1.0 if p == 0 else -1.0)
        reach = np.zeros(self.num_nodes)
        reach[0] = 1.0
        num = np.zeros((self.num_infosets, self.num_actions))
        den = np.zeros(self.num_infosets)
        for i in range(self.num_nodes):
            if self.kind[i] == CHANCE:
                k = self.chance_child[i] >= 0
                reach[self.chance_child[i, k]] = reach[i] * self.chance_prob[i, k]
            elif self.kind[i] == DECISION:
                a = self.legal[i]
                kids = self.child[i, a]
                if self.player[i] == p:
                    reach[kids] = reach[i]
                    num[self.info[i], a] += reach[i] * (v[kids] - v[i])
                    den[self.info[i]] += reach[i]
                else:
                    reach[kids] = reach[i] * sigma[self.info[i], a]
        return num / np.maximum(den, 1e-300)[:, None]


def sample_episodes(tree, sigma, q, traverser, n, epsilon, rng, scale=1.0):
    """``n`` outcome-sampling episodes for ``traverser``, all at once.  The traverser samples
    ``xi = eps * uniform + (1 - eps) * sigma``, the opponent ``sigma``, chance its distribution.  Backwards along
    each trajectory, at every decision node h with sampled action a* and child h':

        v(h, a) = Q_i(h, a) + [a = a*] (v(h') - Q_i(h, a)) / xi(a*),      v(h) = sum_a sigma(a) v(h, a)

    with Q_i = +-q (the baseline of player 0's value) and terminal values u_i / scale.  Traverser nodes give an
    advantage sample v(h, .) - v(h); opponent nodes a strategy sample; every decision a baseline transition."""
    sign = 1.0 if traverser == 0 else -1.0
    visited = 0

    def deal(nodes):  # resolve chance nodes (several can follow each other)
        nonlocal visited
        while True:
            c = np.flatnonzero(tree.kind[nodes] == CHANCE)
            if len(c) == 0:
                return nodes
            j = (rng.random(len(c))[:, None] >= tree.chance_cum[nodes[c]]).sum(1)
            nodes[c] = tree.chance_child[nodes[c], j]
            visited += len(c)

    rows = np.arange(n)  # the episodes still running
    h = deal(np.zeros(n, dtype=np.int64))  # their current decision nodes
    value = np.zeros(n)  # per episode: the value of the node below the step being processed
    reach = np.ones(n)  # the traverser's own sampling reach of the current node
    steps = []
    while len(rows):
        visited += len(rows)
        info, legal = tree.info[h], tree.legal[h]
        sig = sigma[info]
        mine = tree.player[h] == traverser
        xi = np.where(mine[:, None], epsilon * legal / legal.sum(1, keepdims=True) + (1.0 - epsilon) * sig, sig)
        cum = np.cumsum(xi, axis=1)
        # inverse CDF: the number of cumulative sums not above the draw; an action with probability 0 is never hit
        a = (rng.random(len(rows))[:, None] * cum[:, -1:] >= cum).sum(1)
        ar = np.arange(len(rows))
        p_a = xi[ar, a] / cum[:, -1]
        nxt = deal(tree.child[h, a])
        done = tree.kind[nxt] == TERMINAL
        visited += int(done.sum())
        value[rows[done]] = sign * tree.util[nxt[done]] / scale
        steps.append(dict(rows=rows, h=h, info=info, legal=legal, sig=sig, mine=mine, a=a, p=p_a, nxt=nxt, done=done, reach=reach[rows]))
        reach[rows[mine]] *= p_a[mine]
        rows, h = rows[~done], nxt[~done]

    adv_info, adv, adv_reach, strat_info = [], [], [], []
    for s in reversed(steps):
        ar = np.arange(len(s["rows"]))
        qh = sign * q[tree.dec[s["h"]]] * s["legal"]
        qh[ar, s["a"]] += (value[s["rows"]] - qh[ar, s["a"]]) / s["p"]
        v = (qh * s["sig"]).sum(1)
        mine = s["mine"]
        adv_info.append(s["info"][mine])
        adv.append(((qh - v[:, None]) * s["legal"])[mine])
        adv_reach.append(s["reach"][mine])
        strat_info.append(s["info"][~mine])
        value[s["rows"]] = v
    cat = np.concatenate
    return {
        "adv_info": cat(adv_info), "adv": cat(adv), "adv_reach": cat(adv_reach), "strat_info": cat(strat_info),
        "q_node": cat([tree.dec[s["h"]] for s in steps]),
        "q_action": cat([s["a"] for s in steps]),
        "q_next": cat([np.where(s["done"], 0, tree.dec[s["nxt"]]) for s in steps]),
        "q_next_info": cat([np.where(s["done"], 0, tree.info[s["nxt"]]) for s in steps]),
        "q_reward": cat([np.where(s["done"], tree.util[s["nxt"]] / scale, 0.0) for s in steps]),
        "q_done": cat([s["done"].astype(np.float32) for s in steps]),
        "nodes": visited,
    }
