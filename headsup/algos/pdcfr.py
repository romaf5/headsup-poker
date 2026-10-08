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


class Transitions:
    """Circular buffer of baseline transitions, stored as decision-node / infoset indices."""

    FIELDS = (("node", np.int64), ("action", np.int64), ("next", np.int64), ("next_info", np.int64),
              ("reward", np.float32), ("done", np.float32))

    def __init__(self, capacity):
        self.capacity, self.pos, self.size = int(capacity), 0, 0
        self.data = {k: np.zeros(self.capacity, dtype=t) for k, t in self.FIELDS}

    def add(self, **cols):
        n = len(cols["node"])
        idx = (self.pos + np.arange(n)) % self.capacity
        for k, _ in self.FIELDS:
            self.data[k][idx] = cols[k]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)


def advantage_target(frozen_out, legal, d, adv):
    """The regression target of the cumulative-advantage net: the previous net's output clipped at zero (the
    "+" of DCFR+, applied where the net is read), discounted, plus this iteration's sampled advantages."""
    return torch.clamp(frozen_out * legal, min=0.0) * d + adv


def baseline_target(reward, done, next_q, next_sigma):
    """Expected SARSA for the baseline: the reward plus the next state's values under the next strategy."""
    return reward + (1.0 - done) * (next_q * next_sigma).sum(1)


class PDCFRSolver:
    def __init__(self, game, variant="pdcfr+", traversals=10_000, epsilon=0.6, alpha=None, gamma=None, discount_offset=None,
                 adv_steps=750, adv_batch=2048, q_steps=1000, q_batch=2048, q_capacity=1_000_000, q_sync=50,
                 policy_steps=5000, policy_batch=2048, strat_capacity=1_000_000, lr=1e-3, hidden=64, layers=3,
                 fallback="authors", reinit_prediction=False, baseline=True, reach_weighted=False, device="cpu", seed=0):
        assert variant in VARIANTS and fallback in ("authors", "argmax", "uniform")
        self.game, self.variant = game, variant
        d = DEFAULTS[variant]
        self.alpha = d["alpha"] if alpha is None else alpha
        self.gamma = d["gamma"] if gamma is None else gamma
        self.offset = d["offset"] if discount_offset is None else discount_offset
        self.traversals, self.epsilon, self.lr = traversals, epsilon, lr
        self.adv_steps, self.adv_batch = adv_steps, adv_batch
        self.q_steps, self.q_batch, self.q_sync = q_steps, q_batch, q_sync
        self.policy_steps, self.policy_batch = policy_steps, policy_batch
        self.hidden, self.layers = hidden, layers
        self.fallback, self.reinit_prediction = fallback, reinit_prediction
        self.baseline, self.reach_weighted = baseline, reach_weighted
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.tree = tree = Tree(game)
        self.scale = tree.max_utility  # utilities are normalised to [-1, 1]
        A, D = game.num_actions, game.obs_dim
        self.info_obs_t = torch.as_tensor(tree.info_obs, device=self.device)
        self.info_legal_t = torch.as_tensor(tree.info_legal, dtype=torch.float32, device=self.device)
        self.hist_x_t = torch.as_tensor(tree.hist_x, device=self.device)
        # persistent networks and optimisers: they are never re-initialised (unless ``reinit_prediction``)
        self.R = [self._net(D) for _ in range(2)]
        self.opt_R = [_adam(n, lr) for n in self.R]
        self.r = [self._net(D) for _ in range(2)] if variant == "pdcfr+" else None
        self.opt_r = [_adam(n, lr) for n in self.r] if self.r else None
        self.q_state = None  # the baseline net of the last fit (re-created at every fit), for checkpoints
        self.q_tab = np.zeros((tree.num_decisions, A))  # player 0's baseline values at every decision node
        self.q_memory = Transitions(q_capacity)
        self.strat_memory = ReservoirBuffer(strat_capacity, self.device, obs_dim=D, target_dim=A, seed=seed + 2, int_dim=0, legal_dim=A)
        self.iteration = self.episodes = self.nodes_touched = 0
        self.sigma = self._strategy_table(0.0)

    def _net(self, in_dim):
        return mlp(in_dim, self.game.num_actions, self.hidden, self.layers).to(self.device).eval()

    def _strategy_table(self, d):
        """Both players' strategies at every infoset from the current networks, with discount ``d``."""
        with torch.no_grad():
            R = torch.stack([n(self.info_obs_t) for n in self.R]).cpu().numpy().astype(np.float64)
            r = torch.stack([n(self.info_obs_t) for n in self.r]).cpu().numpy().astype(np.float64) if self.r else R
        own = self.tree.info_player
        rows = np.arange(len(own))
        return strategy_rows(R[own, rows], r[own, rows], self.tree.info_legal, d, self.variant, self.fallback)

    # -- fits ----------------------------------------------------------------------------------------
    def _fit_advantage(self, p, data, d):
        m = len(data["adv_info"])
        if m == 0:
            return
        idx = torch.as_tensor(data["adv_info"], device=self.device)
        adv = data["adv"] / data["adv_reach"][:, None] if self.reach_weighted else data["adv"]  # the paper's "w/o adv" ablation
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        legal, obs, R = self.info_legal_t, self.info_obs_t, self.R[p]
        with torch.no_grad():  # the network as it is before this fit, at every infoset
            frozen = R(obs)
        nets, opts = [R], [self.opt_R[p]]
        r = None
        if self.r:
            if self.reinit_prediction:  # the paper's table; the authors' code never re-initialises it
                self.r[p] = self._net(self.game.obs_dim)
                self.opt_r[p] = _adam(self.r[p], self.lr)
            r = self.r[p]
            nets.append(r)
            opts.append(self.opt_r[p])

        def loss_fn():
            j = _batch_index(m, self.adv_batch, self.device)
            i = idx[j]
            x, lg = obs[i], legal[i]
            loss = (R(x) * lg - advantage_target(frozen[i], lg, d, adv[j])).pow(2).mean()
            if r is not None:  # the same minibatch
                loss = loss + (r(x) * lg - adv[j]).pow(2).mean()
            return loss

        _optimise(nets, opts, loss_fn, self.adv_steps, grad_clip=0)

    def _fit_baseline(self, t):
        mem = self.q_memory
        n = mem.size
        if n == 0:
            return
        q, target = self._net(2 * self.game.obs_dim), self._net(2 * self.game.obs_dim)
        target.load_state_dict(q.state_dict())
        # the strategies the next iteration will play, from the networks just updated
        sigma_next = torch.as_tensor(self._strategy_table(discount(t + 1, self.alpha, self.offset)), dtype=torch.float32, device=self.device)
        col = {k: torch.as_tensor(v[:n], device=self.device) for k, v in mem.data.items()}
        hist = self.hist_x_t

        def loss_fn():
            j = _batch_index(n, self.q_batch, self.device)
            pred = q(hist[col["node"][j]]).gather(1, col["action"][j][:, None]).squeeze(1)
            with torch.no_grad():
                tgt = baseline_target(col["reward"][j], col["done"][j], target(hist[col["next"][j]]), sigma_next[col["next_info"][j]])
            return (pred - tgt).pow(2).mean()

        _optimise(q, _adam(q, self.lr), loss_fn, self.q_steps, grad_clip=0, sync_every=self.q_sync,
                  sync_fn=lambda: target.load_state_dict(q.state_dict()))
        with torch.no_grad():
            self.q_tab = q(hist).cpu().numpy().astype(np.float64)
        self.q_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}

    # -- the iteration ---------------------------------------------------------------------------------
    def iterate(self, n=1):
        tree = self.tree
        for _ in range(n):
            self.iteration += 1
            t = self.iteration
            d = discount(t, self.alpha, self.offset)
            self.sigma = self._strategy_table(d)
            for p in (0, 1):  # alternating: player 1's episodes already use player 0's updated networks
                data = sample_episodes(tree, self.sigma, self.q_tab, p, self.traversals, self.epsilon, self.rng, self.scale)
                self.episodes += self.traversals
                self.nodes_touched += data["nodes"]
                k = data["strat_info"]
                if len(k):
                    self.strat_memory.add(tree.info_obs[k], np.full(len(k), t, np.float32), self.sigma[k], tree.info_legal[k])
                if self.baseline:
                    self.q_memory.add(node=data["q_node"], action=data["q_action"], next=data["q_next"],
                                      next_info=data["q_next_info"], reward=data["q_reward"], done=data["q_done"])
                self._fit_advantage(p, data, d)
                self.sigma = self._strategy_table(d)
                if self.baseline:
                    self._fit_baseline(t)
        return self

    # -- policies and evaluation ---------------------------------------------------------------------------
    def _tabular(self, probs):
        return TabularPolicy(self.game, {k: probs[i] for i, k in enumerate(self.tree.info_keys)})

    def current_policy(self):
        """The strategy the next iteration would play."""
        return self._tabular(self._strategy_table(discount(self.iteration + 1, self.alpha, self.offset)))

    def policy_net(self):
        """The average-strategy network, fitted from scratch on the strategy reservoir with weights (2 t / T)^gamma."""
        net = self._net(self.game.obs_dim)
        buf = self.strat_memory
        n = len(buf)
        if n == 0:
            return net
        scale, gamma = 2.0 / max(self.iteration, 1), self.gamma

        def loss_fn():
            i = _batch_index(n, self.policy_batch, self.device)
            logits = net(buf.obs_float[i])
            probs = torch.softmax(torch.where(buf.legal[i] > 0, logits, torch.full_like(logits, -1e20)), dim=-1)
            w = (buf.t[i] * scale).pow(gamma)
            return (w[:, None] * (probs - buf.target[i]).pow(2)).mean()

        _optimise(net, _adam(net, self.lr), loss_fn, self.policy_steps, grad_clip=0)
        return net

    def average_policy(self):
        net = self.policy_net()
        with torch.no_grad():
            logits = net(self.info_obs_t)
            probs = torch.softmax(torch.where(self.info_legal_t > 0, logits, torch.full_like(logits, -1e20)), dim=-1)
        return self._tabular(probs.cpu().numpy().astype(np.float64))

    def evaluate(self):
        return {"current": exploitability(self.game, self.current_policy())[0],
                "average": exploitability(self.game, self.average_policy())[0]}

    # -- checkpoints -----------------------------------------------------------------------------------
    def state_dict(self):
        cpu = lambda net: {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}  # noqa: E731
        mem = self.q_memory
        return {
            "variant": self.variant, "iteration": self.iteration, "episodes": self.episodes, "nodes_touched": self.nodes_touched,
            "R": [cpu(n) for n in self.R], "opt_R": [o.state_dict() for o in self.opt_R],
            "r": [cpu(n) for n in self.r] if self.r else None, "opt_r": [o.state_dict() for o in self.opt_r] if self.r else None,
            "q_state": self.q_state, "q_tab": self.q_tab.copy(), "sigma": self.sigma.copy(),
            "q_memory": {"pos": mem.pos, "size": mem.size, **{k: v[: mem.size].copy() for k, v in mem.data.items()}},
            "strat_memory": self.strat_memory.state_dict(), "strat_rng": self.strat_memory.rng.bit_generator.state,
            "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        if state["variant"] != self.variant:
            raise ValueError(f"checkpoint of variant {state['variant']!r} cannot be loaded into a {self.variant!r} solver")
        self.iteration, self.episodes, self.nodes_touched = int(state["iteration"]), int(state["episodes"]), int(state["nodes_touched"])
        for net, opt, sd, osd in zip(self.R, self.opt_R, state["R"], state["opt_R"]):
            net.load_state_dict(sd)
            opt.load_state_dict(osd)
        if self.r:
            for net, opt, sd, osd in zip(self.r, self.opt_r, state["r"], state["opt_r"]):
                net.load_state_dict(sd)
                opt.load_state_dict(osd)
        self.q_state, self.q_tab, self.sigma = state["q_state"], state["q_tab"].copy(), state["sigma"].copy()
        mem, saved = self.q_memory, state["q_memory"]
        mem.pos, mem.size = int(saved["pos"]), int(saved["size"])
        for k, _ in Transitions.FIELDS:
            mem.data[k][: mem.size] = saved[k]
        self.strat_memory.load_state_dict(state["strat_memory"])
        self.strat_memory.rng.bit_generator.state = state["strat_rng"]
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        return self


def main(argv=None):
    from headsup.games import make_game

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--variant", default="pdcfr+", choices=VARIANTS, help="dcfr+ = VR-DeepDCFR+, pdcfr+ = VR-DeepPDCFR+")
    p.add_argument("--episodes", type=int, default=10_000_000, help="sampled episodes in total; iterations = episodes // (2 x traversals)")
    p.add_argument("--traversals", type=int, default=10_000, help="episodes per player per iteration")
    p.add_argument("--epsilon", type=float, default=0.6, help="the traverser's exploration")
    p.add_argument("--alpha", type=float, default=None, help="regret discount exponent (default: 2 for dcfr+, 2.3 for pdcfr+)")
    p.add_argument("--gamma", type=float, default=None, help="average-strategy weight exponent (default 2)")
    p.add_argument("--discount-offset", type=float, default=None,
                   help="constant in the discount's denominator (default: the authors' code - 1.5 for dcfr+, 1 for pdcfr+; the paper: 1)")
    p.add_argument("--adv-steps", type=int, default=750)
    p.add_argument("--adv-batch", type=int, default=2048)
    p.add_argument("--q-steps", type=int, default=1000, help="baseline fit steps (the authors' configs; the paper's table says 10000)")
    p.add_argument("--q-batch", type=int, default=2048)
    p.add_argument("--policy-steps", type=int, default=5000)
    p.add_argument("--policy-batch", type=int, default=2048)
    p.add_argument("--fallback", default="authors", choices=["authors", "argmax", "uniform"],
                   help="strategy when no (predicted) regret is positive: the authors' code, the largest unclipped score, or uniform")
    p.add_argument("--reinit-prediction", action="store_true", help="re-initialise the prediction net every iteration (the paper's table)")
    p.add_argument("--no-baseline", action="store_true", help="no variance reduction (the paper's DeepPDCFR+ ablation)")
    p.add_argument("--reach-weighted", action="store_true", help="divide the samples by the traverser's sampling reach (the 'w/o adv' ablation)")
    p.add_argument("--eval-every", type=int, default=3, help="fit and evaluate the average policy every this many iterations (and at 1, 2 and the last)")
    p.add_argument("--checkpoint", default=None, help="saved at evaluations; an existing file is resumed from")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0, help="at most one checkpoint per this many minutes (plus the final one)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    game = make_game(args.game)
    solver = PDCFRSolver(game, args.variant, traversals=args.traversals, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma,
                         discount_offset=args.discount_offset, adv_steps=args.adv_steps, adv_batch=args.adv_batch, q_steps=args.q_steps,
                         q_batch=args.q_batch, policy_steps=args.policy_steps, policy_batch=args.policy_batch, fallback=args.fallback,
                         reinit_prediction=args.reinit_prediction, baseline=not args.no_baseline, reach_weighted=args.reach_weighted,
                         device=args.device, seed=args.seed)
    iterations = args.episodes // (2 * args.traversals)
    curve, elapsed = [], 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        solver.load_state_dict(saved["solver"])
        curve, elapsed = saved["curve"], saved["seconds"]
        print(f"resumed from {args.checkpoint} at iteration {solver.iteration}", flush=True)
    t0 = time.perf_counter() - elapsed
    last_save = time.perf_counter()

    def dump():
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"game": args.game, "algo": args.variant, "args": vars(args), "curve": curve}, f, indent=2)

    for it in range(solver.iteration + 1, iterations + 1):
        solver.iterate()
        if it % args.eval_every == 0 or it < args.eval_every or it == iterations:
            ev = solver.evaluate()
            curve.append({"iteration": it, **ev, "nodes_touched": solver.nodes_touched, "episodes": solver.episodes,
                          "seconds": time.perf_counter() - t0})
            print(f"{args.game} {args.variant} it {it}: exploitability current {ev['current']:.4f} average {ev['average']:.4f}  "
                  f"nodes {solver.nodes_touched:.3g}  episodes {solver.episodes}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            dump()
            due = time.perf_counter() - last_save >= 60.0 * args.checkpoint_minutes or it == iterations
            if args.checkpoint and due:  # atomic: a crash while writing leaves the previous checkpoint intact
                torch.save({"solver": solver.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}, args.checkpoint + ".tmp")
                os.replace(args.checkpoint + ".tmp", args.checkpoint)
                last_save = time.perf_counter()
    dump()


if __name__ == "__main__":
    main()
