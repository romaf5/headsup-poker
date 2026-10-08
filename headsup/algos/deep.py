"""Deep CFR family on the game protocol: Deep CFR, Single Deep CFR, DREAM and ESCHER.

One solver, four sample collectors; everything else (reservoir memories, from-scratch network
fits with iteration weights, the bank of iterates, exact evaluation) is shared:

* ``deepcfr`` / ``sdcfr`` (Brown et al. 2019; Steinberger 2019): external-sampling traversals; the
  traverser explores all actions, the opponent samples from regret matching on its advantage net;
  samples ``(s, t, v(a) - sum_a sigma v)`` (advantage memory) and ``(s, t, sigma)`` (strategy
  memory); advantage nets refitted from scratch each iteration with weight t; the average
  strategy is the policy net (deepcfr) or the reach-weighted mixture of all iterates (sdcfr).
* ``dream`` (Steinberger, Lerer & Brown 2020): outcome sampling - the traverser samples with
  xi = eps * uniform + (1 - eps) * sigma, the opponent with sigma; a per-player baseline network
  Q_i(s*(h), a) on the concatenated infostates of all players (their footnote 3), fine-tuned by
  expected SARSA on a circular buffer; baseline-corrected sampled values (their eq. 6-7); the
  advantage sample weight is t / x^xi_i(s_i) (own sampling reach); averaging as SD-CFR.
* ``escher`` (McAleer et al. 2023; as their reference code): the update player samples uniformly
  (a fixed sampling policy), the opponent from sigma; a history value net q(h, a) (both players'
  infostates -> player 0's value per action) re-fitted from scratch every iteration on that
  iteration's self-play trajectories (both players play 0.99 sigma + 0.01 uniform, importance-
  weighted returns); the regret estimate is q_i(h, a) - sum_a sigma_i(s, a) q_i(h, a) with no
  importance weights; cumulative regret buffer -> regret net from scratch each iteration; the
  average policy net is fitted on (I, t, sigma) samples taken at the *opponent's* on-policy
  infosets (as Deep CFR; the update player's uniformly sampled infosets would weight the average
  by the wrong reach).

Networks come from ``game.make_model()``; ``policy_*`` helpers wrap them for the exact
:func:`headsup.algos.best_response.exploitability` check.  Small games only (Python traversal);
the hold'em pipeline in headsup.deepcfr keeps its C++ kernels.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from headsup.algos.best_response import TabularPolicy, exploitability
from headsup.deepcfr.memory import ReservoirBuffer

ALGOS = ("deepcfr", "sdcfr", "dream", "escher")


# ----------------------------------------------------------------------------- small utilities
class CircularBuffer:
    """Fixed-capacity FIFO of (x, target, weight) rows (DREAM's B^q, ESCHER's value data)."""

    def __init__(self, capacity, x_dim, target_dim):
        self.x = np.zeros((capacity, x_dim), np.float32)
        self.target = np.zeros((capacity, target_dim), np.float32)
        self.mask = np.zeros((capacity, target_dim), np.float32)
        self.pos, self.size, self.capacity = 0, 0, capacity

    def add(self, x, target, mask):
        n = len(x)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.x[idx], self.target[idx], self.mask[idx] = x, target, mask
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, n)
        return self.x[idx], self.target[idx], self.mask[idx]


def _adam(model, lr):
    """Adam (lr 1e-3 in all the papers); fused and CUDA-graph capturable on the GPU."""
    cuda = next(model.parameters()).is_cuda
    return torch.optim.Adam(model.parameters(), lr=lr, **({"fused": True, "capturable": True} if cuda else {}))


def _load_optimiser(opt, saved):
    """Load an optimiser's state without sharing tensors with ``saved`` (two solvers must never step one Adam)
    and without taking over the checkpoint's device-dependent switches: fused / capturable are what ``_adam``
    chose for THIS solver's device (a CPU-written checkpoint would leave a CUDA solver's persistent optimiser
    non-capturable inside the CUDA graph)."""
    saved = copy.deepcopy(saved)
    for group, mine in zip(saved["param_groups"], opt.param_groups):
        for key in ("fused", "capturable", "foreach"):
            group[key] = mine.get(key)
    opt.load_state_dict(saved)


_SIDE_STREAMS = {}


def _side_stream(device):
    """One warm-up stream per device: PyTorch keeps a cuBLAS workspace per stream it has seen, so a
    new stream per fit leaked ~20 MB each time (11 Leduc runs ran out of GPU memory in 20 minutes)."""
    if device not in _SIDE_STREAMS:
        _SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return _SIDE_STREAMS[device]


def _optimise(model, opt, loss_fn, steps, grad_clip=1.0, sync_every=0, sync_fn=None):
    """``steps`` optimiser steps on ``loss_fn()`` (which samples its own minibatch on the model's
    device), gradient-norm clipping as the papers.  On CUDA the whole step - sampling, forward,
    backward, clipping, fused Adam - is captured once in a CUDA graph and replayed: the small games'
    networks are launch-bound (5 ms -> ~0.3-0.7 ms per step).  ``model`` / ``opt`` may be lists (several
    networks with disjoint parameters stepped on one loss); ``sync_fn`` is called after step 1 and then
    after every ``sync_every``-th step (a target-network copy: in-place, so the graph sees it).
    Returns the last loss."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    opts = list(opt) if isinstance(opt, (list, tuple)) else [opt]
    params = [p for m in models for p in m.parameters()]
    cuda = params[0].is_cuda

    def step():
        loss = loss_fn()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip, foreach=cuda)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=not cuda)  # the graph needs the gradient buffers to stay in place
        return loss

    def after(done):
        if sync_every and (done - 1) % sync_every == 0:
            sync_fn()

    for m in models:
        m.train()
    loss = None
    if not cuda or steps <= 4:
        for i in range(steps):
            loss = step()
            after(i + 1)
    else:
        side = _side_stream(params[0].device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up outside the graph (allocates the gradient / optimiser state)
            for i in range(3):
                step()
                after(i + 1)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # recorded, not executed
            loss = step()
        for i in range(steps - 3):
            graph.replay()
            after(i + 4)
    for m in models:
        m.eval()
    return float(loss.item()) if loss is not None else float("nan")


def _batch_index(n, batch, device):
    return (torch.rand(batch, device=device) * n).long()  # uniform with replacement (graph-safe RNG)


def _fit(model, opt, xs, targets, weights, steps, batch, device, masks=None, grad_clip=1.0):
    """Weighted MSE regression steps on numpy arrays; ``masks`` restricts the loss to some outputs."""
    xs_t = torch.as_tensor(xs, device=device)
    tg_t = torch.as_tensor(targets, device=device)
    w_t = torch.as_tensor(weights, device=device)
    m_t = torch.as_tensor(masks, device=device) if masks is not None else None
    n = len(xs)

    def loss_fn():
        idx = _batch_index(n, batch, device)
        err = (model(xs_t[idx]) - tg_t[idx]).pow(2)
        if m_t is not None:
            err = err * m_t[idx]
        return (w_t[idx][:, None] * err).mean()

    return _optimise(model, opt, loss_fn, steps, grad_clip)


def _fit_from_buffer(model, buffer, steps, batch, lr, device, loss="mse", weight_power=1.0, weight_scale=1.0, grad_clip=1.0, legal_fn=None):
    """DeepCFR-style fit on a ReservoirBuffer (on ``device``): targets weighted by t^power * ``weight_scale``
    (the SD-CFR authors' code divides by the latest iteration, keeping the loss O(1)); ``loss`` mse
    (advantages) or 'policy' (softmax(logits) vs stored probabilities)."""
    n = len(buffer)
    if n == 0:
        return model

    def loss_fn():
        idx = _batch_index(n, batch, device)
        obs = torch.cat([buffer.obs_int[idx].to(torch.float32), buffer.obs_float[idx]], dim=1)
        pred = model(obs)
        if loss == "policy":
            pred = torch.softmax(pred, dim=-1)
        err = (pred - buffer.target[idx]).pow(2)
        if legal_fn is not None:  # the loss on legal actions only (illegal outputs are never used)
            err = err * legal_fn(obs)
        return (weight_scale * buffer.t[idx].pow(weight_power)[:, None] * err).mean()

    _optimise(model, _adam(model, lr), loss_fn, steps, grad_clip)
    return model


def regret_matching_rows(adv, legal, argmax_fallback=False):
    """Row-wise :func:`regret_matching_np` for (..., A) arrays."""
    pos = np.where(legal, np.maximum(adv, 0.0), 0.0)
    total = pos.sum(-1, keepdims=True)
    if argmax_fallback:
        fb = np.zeros_like(pos)
        np.put_along_axis(fb, np.argmax(np.where(legal, adv, -np.inf), axis=-1)[..., None], 1.0, axis=-1)
    else:
        fb = legal / legal.sum(-1, keepdims=True)
    return np.where(total > 1e-12, pos / np.maximum(total, 1e-300), fb)


def regret_matching_np(adv, legal, argmax_fallback=False):
    pos = np.where(legal, np.maximum(adv, 0.0), 0.0)
    total = pos.sum()
    if total > 1e-12:
        return pos / total
    if argmax_fallback:
        p = np.zeros_like(pos)
        p[int(np.argmax(np.where(legal, adv, -np.inf)))] = 1.0
        return p
    p = legal.astype(np.float64)
    return p / p.sum()


class NetPolicy:
    """policy(state) = regret matching (or softmax) on a network's output; caches per infoset."""

    def __init__(self, game, nets, mode="rm", device="cpu", argmax=False):
        self.game, self.nets, self.mode, self.device = game, nets, mode, device
        self.argmax = argmax  # regret matching's fallback where no advantage is positive: the best action / uniform
        self.cache = {}

    def __call__(self, state):
        p = state.current_player
        key = state.info_key(p)
        probs = self.cache.get(key)
        if probs is None:
            net = self.nets[p] if isinstance(self.nets, (list, tuple)) else self.nets
            with torch.no_grad():
                out = net(torch.as_tensor(state.info_state(p), device=self.device)[None])[0].cpu().numpy().astype(np.float64)
            legal = state.legal_mask()
            if self.mode == "rm":
                probs = regret_matching_np(out, legal, self.argmax)
            else:
                out = np.where(legal, out, -np.inf)
                e = np.exp(out - out.max())
                probs = e / e.sum()
            self.cache[key] = probs
        return probs


# ----------------------------------------------------------------------------- the solver
class DeepSolver:
    def __init__(self, game, algo="deepcfr", traversals=1000, adv_capacity=2_000_000, strat_capacity=2_000_000,
                 adv_steps=3000, adv_batch=2048, policy_steps=4000, policy_batch=2048, q_steps=1000, q_batch=512,
                 q_capacity=200_000, value_traversals=None, epsilon=0.5, value_epsilon=0.01, lr=1e-3, device="cpu", seed=0,
                 model_kwargs=None, rm_argmax=True, warm_start=False, normalized_weights=False, grad_clip=1.0, mean_regret=False,
                 masked_loss=False, shared_baseline=False, bootstrap_chance=False):
        assert algo in ALGOS
        self.game, self.algo = game, algo
        self.traversals = traversals
        self.value_traversals = value_traversals or traversals
        self.adv_steps, self.adv_batch = adv_steps, adv_batch
        self.policy_steps, self.policy_batch = policy_steps, policy_batch
        self.q_steps, self.q_batch = q_steps, q_batch
        self.epsilon, self.value_epsilon, self.lr = epsilon, value_epsilon, lr
        self.rm_argmax = rm_argmax
        self.warm_start = warm_start  # SD-CFR paper (Leduc): each advantage net starts from the player's previous one
        # the SD-CFR authors' code: loss weights t / t_latest (O(1) losses), gradient clipping at 10, and the
        # sampled regrets divided by the number of legal actions (their multi-outcome sampler's mean)
        self.normalized_weights, self.grad_clip, self.mean_regret = normalized_weights, grad_clip, mean_regret
        self.device = torch.device(device)
        # the advantage loss on legal actions only - what the authors' masked network outputs amount to in training
        self._legal_fn = None
        if masked_loss:
            from headsup.games.leduc import legal_mask_from_info_state

            caps = torch.tensor(game.max_raises, dtype=torch.float32, device=self.device)
            self._legal_fn = lambda x: legal_mask_from_info_state(game, x, caps)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.model_kwargs = model_kwargs or {}
        A, D = game.num_actions, game.obs_dim
        self.nets = [self._new_model() for _ in range(2)]
        self.iterates = [[self._cpu_state(n)] for n in self.nets]
        self.adv_memory = [ReservoirBuffer(adv_capacity, self.device, obs_dim=D, target_dim=A, seed=seed + i, int_dim=0) for i in range(2)]
        # the average-strategy memory is only needed where a policy net is fitted (SD-CFR / DREAM average the iterates)
        self.strat_memory = ReservoirBuffer(strat_capacity if algo in ("deepcfr", "escher") else 1, self.device, obs_dim=D, target_dim=A,
                                            seed=seed + 2, int_dim=0)
        # DREAM's baseline: Q_i(s*(h), a) per player, expected-SARSA targets; with ``shared_baseline`` one net of player 0's
        # values (negated for player 1) trained once per iteration on both traversers' data (the DREAM authors' code);
        # with ``bootstrap_chance`` the target of a transition followed by a deal bootstraps from the baseline after the
        # sampled deal (their env deals inside step) instead of using the sampled, importance-weighted continuation
        self.shared_baseline, self.bootstrap_chance = shared_baseline, bootstrap_chance
        if algo == "dream":
            n_q = 1 if shared_baseline else 2
            self.q_nets = [self._new_model(2 * D) for _ in range(n_q)]
            self.q_opts = [_adam(n, lr) for n in self.q_nets]
            self.q_memory = [CircularBuffer(q_capacity, 2 * D, A) for _ in range(n_q)]
        if algo == "escher":  # history value q(h, a): player 0's expected return per action
            self.v_net = self._new_model(2 * D)
        self.iteration = 0
        self.nodes_touched = 0  # states visited by the traversals / trajectories (the DREAM paper's x-axis)
        self.stats = {}
        # small games: every infoset / history is enumerated once and the networks are evaluated in
        # one batch per fit (exact tables) instead of one forward pass per visited state
        self._enumerate()
        self._sigma_tab = [None, None]
        self._q_tab = [None, None]
        self._v_tab = None

    def _enumerate(self):
        info, hist = {}, {}

        def walk(state):
            if state.is_terminal():
                return
            if state.is_chance():
                for a, _ in state.chance_outcomes():
                    walk(state.child(a))
                return
            p = state.current_player
            k = state.info_key(p)
            if k not in info:
                info[k] = (p, state.info_state(p), state.legal_mask())
            hk = state.history_key()
            if hk not in hist:
                hist[hk] = self._history(state)
            for a in state.legal_actions():
                walk(state.child(a))

        walk(self.game.new_initial_state())
        self._info_keys = list(info)
        self._info_index = {k: i for i, k in enumerate(self._info_keys)}
        self._info_player = np.array([info[k][0] for k in self._info_keys])
        self._info_obs = np.stack([info[k][1] for k in self._info_keys]).astype(np.float32)
        self._info_legal = np.stack([info[k][2] for k in self._info_keys])
        self._hist_keys = list(hist)
        self._hist_x = np.stack([hist[k] for k in self._hist_keys]).astype(np.float32)

    def _forward(self, net, x):
        with torch.no_grad():
            out = []
            for i in range(0, len(x), 65536):
                out.append(net(torch.as_tensor(x[i : i + 65536], device=self.device)).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def _refresh_sigma(self, p):
        """Regret-matching strategy of player p's current net at every infoset (a dict)."""
        out = self._forward(self.nets[p], self._info_obs)
        tab = {}
        for i, k in enumerate(self._info_keys):
            if self._info_player[i] == p:
                tab[k] = regret_matching_np(out[i], self._info_legal[i], self.rm_argmax)
        self._sigma_tab[p] = tab

    def _refresh_q(self, p):
        out = self._forward(self.q_nets[p], self._hist_x)
        self._q_tab[p] = {k: out[i] for i, k in enumerate(self._hist_keys)}

    def _refresh_v(self):
        out = self._forward(self.v_net, self._hist_x)
        self._v_tab = {k: out[i] for i, k in enumerate(self._hist_keys)}

    # -- helpers ----------------------------------------------------------------------------
    def _new_model(self, in_dim=None, policy=False):
        arch = str(self.model_kwargs.get("arch", ""))
        kw = dict(self.model_kwargs, **({"policy": True} if policy and (arch.startswith("pokerrl") or arch == "deepcfr_dueling") else {}))
        m = self.game.make_model(**kw) if in_dim is None else self.game.make_model(in_dim=in_dim, **kw)
        return m.to(self.device).eval()

    @staticmethod
    def _cpu_state(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _sigma(self, p, state):
        """Regret matching on player p's current advantage net at ``state`` (tabulated)."""
        if self._sigma_tab[p] is None:
            self._refresh_sigma(p)
        return self._sigma_tab[p][state.info_key(p)]

    @staticmethod
    def _history(state):
        return np.concatenate([state.info_state(0), state.info_state(1)])

    # -- external sampling (Deep CFR / SD-CFR) --------------------------------------------------
    def _es(self, state, p, t, adv, strat):
        self.nodes_touched += 1
        if state.is_terminal():
            return state.returns()[p]
        if state.is_chance():
            return self._es(state.child(state.sample_chance(self.rng)), p, t, adv, strat)
        cur = state.current_player
        sigma = self._sigma(cur, state)
        if cur != p:
            strat.append((state.info_state(cur), t, sigma))
            a = int(self.rng.choice(len(sigma), p=sigma))
            return self._es(state.child(a), p, t, adv, strat)
        legal = state.legal_mask()
        values = np.zeros(self.game.num_actions)
        for a in np.flatnonzero(legal):
            values[a] = self._es(state.child(a), p, t, adv, strat)
        v = float(sigma @ values)
        r = np.where(legal, values - v, 0.0)
        adv.append((state.info_state(p), t, r / legal.sum() if self.mean_regret else r))
        return v

    # -- outcome sampling with baselines (DREAM) --------------------------------------------------
    def _q(self, p, state):
        i = 0 if self.shared_baseline else p
        if self._q_tab[i] is None:
            self._refresh_q(i)
        q = self._q_tab[i][state.history_key()]
        return -q if self.shared_baseline and p == 1 else q

    def _os_dream(self, state, p, t, own_sample_reach, adv, q_data):
        """Returns the baseline-corrected sampled value of ``state`` for player p (DREAM eq. 6-7)."""
        self.nodes_touched += 1
        if state.is_terminal():
            return state.returns()[p]
        if state.is_chance():
            return self._os_dream(state.child(state.sample_chance(self.rng)), p, t, own_sample_reach, adv, q_data)
        cur = state.current_player
        legal = state.legal_mask()
        sigma = self._sigma(cur, state)
        if cur == p:
            xi = self.epsilon * legal / legal.sum() + (1 - self.epsilon) * sigma
        else:
            xi = sigma
        a = int(self.rng.choice(len(xi), p=xi / xi.sum()))
        hist = self._history(state)
        b = self._q(p, state)  # baseline for every action of the actor at h
        child = state.child(a)
        while self.bootstrap_chance and child.is_chance():  # the deal belongs to this transition (counted as before)
            self.nodes_touched += 1
            child = child.child(child.sample_chance(self.rng))
        v_child = self._os_dream(child, p, t, own_sample_reach * (xi[a] if cur == p else 1.0), adv, q_data)
        va = np.where(legal, b, 0.0)
        va[a] = b[a] + (v_child - b[a]) / xi[a]
        v = float(sigma @ va)
        if cur == p:
            adv.append((state.info_state(p), t / max(own_sample_reach, 1e-12), np.where(legal, va - v, 0.0)))
        # expected SARSA target for Q_p(h, a): reward at h' plus the expected baseline of h' under sigma
        if child.is_terminal():
            target = child.returns()[p]
        elif child.is_chance() or child.current_player < 0:
            target = v_child  # (chance nodes: no baseline, use the sampled continuation)
        else:
            sig_c = self._sigma(child.current_player, child)
            target = float(sig_c @ np.where(child.legal_mask(), self._q(p, child), 0.0))
        row_mask = np.zeros(self.game.num_actions, np.float32)
        row_mask[a] = 1.0
        row_target = np.zeros(self.game.num_actions, np.float32)
        row_target[a] = -target if self.shared_baseline and p == 1 else target  # shared: player 0's values
        q_data.append((hist, row_target, row_mask))
        return v

    # -- ESCHER -------------------------------------------------------------------------------
    def _v(self, state):
        if self._v_tab is None:
            self._refresh_v()
        return self._v_tab[state.history_key()]

    def _escher_value_data(self, data):
        """One trajectory with both players on (1 - e) sigma + e uniform (ESCHER's value exploration,
        e = ``value_epsilon``); rows (h, a, player 0's return after h a importance-weighted by
        sigma / sampling policy along the rest of the trajectory) - unbiased for q_sigma(h, a)."""
        state = self.game.new_initial_state()
        rows = []
        while not state.is_terminal():
            self.nodes_touched += 1
            if state.is_chance():
                state = state.child(state.sample_chance(self.rng))
                continue
            legal = state.legal_mask()
            sigma = self._sigma(state.current_player, state)
            xi = (1.0 - self.value_epsilon) * sigma + self.value_epsilon * legal / legal.sum()
            a = int(self.rng.choice(len(xi), p=xi / xi.sum()))
            rows.append((self._history(state), a, sigma[a] / xi[a]))
            state = state.child(a)
        self.nodes_touched += 1  # the terminal state (counted by the other samplers as well)
        ret = state.returns()[0]
        for hist, a, ratio in reversed(rows):
            tgt = np.zeros(self.game.num_actions, np.float32)
            msk = np.zeros(self.game.num_actions, np.float32)
            tgt[a], msk[a] = ret, 1.0
            data.append((hist, tgt, msk))
            ret *= ratio  # the ratio at h belongs to the returns of the histories before h

    def _escher_regrets(self, p, t, adv, strat):
        """One trajectory: update player p samples uniformly, the opponent from sigma; regrets from q."""
        state = self.game.new_initial_state()
        while not state.is_terminal():
            self.nodes_touched += 1
            if state.is_chance():
                state = state.child(state.sample_chance(self.rng))
                continue
            cur = state.current_player
            legal = state.legal_mask()
            sigma = self._sigma(cur, state)
            if cur == p:
                q = self._v(state) * (1.0 if p == 0 else -1.0)  # player p's value per action
                v = float(sigma @ np.where(legal, q, 0.0))
                adv.append((state.info_state(p), t, np.where(legal, q - v, 0.0)))
                a = int(self.rng.choice(np.flatnonzero(legal)))
            else:
                strat.append((state.info_state(cur), t, sigma))  # on-policy: visited in proportion to its own reach
                a = int(self.rng.choice(len(sigma), p=sigma))
            state = state.child(a)
        self.nodes_touched += 1  # the terminal state

    # -- one iteration --------------------------------------------------------------------------
    def iterate(self, n=1):
        for _ in range(n):
            self.iteration += 1
            t = float(self.iteration)
            t0 = time.perf_counter()
            if self.algo == "escher":  # 1. a fresh history value net on this iteration's trajectories
                data = []
                for _ in range(self.value_traversals):
                    self._escher_value_data(data)
                xs, tgs, mks = (np.stack([d[i] for d in data]).astype(np.float32) for i in range(3))
                self.v_net = self._new_model(2 * self.game.obs_dim).train()
                _fit(self.v_net, _adam(self.v_net, self.lr), xs, tgs, np.ones(len(xs), np.float32), self.q_steps, self.q_batch, self.device, masks=mks)
                self.v_net.eval()
                self._v_tab = None
            for p in range(2):
                adv, strat, q_data = [], [], []
                for _ in range(self.traversals):
                    if self.algo in ("deepcfr", "sdcfr"):
                        self._es(self.game.new_initial_state(), p, t, adv, strat)
                    elif self.algo == "dream":
                        self._os_dream(self.game.new_initial_state(), p, t, 1.0, adv, q_data)
                    else:
                        self._escher_regrets(p, t, adv, strat)
                if adv:
                    self.adv_memory[p].add(np.stack([a[0] for a in adv]), np.array([a[1] for a in adv], np.float32), np.stack([a[2] for a in adv]))
                if strat and self.algo in ("deepcfr", "escher"):
                    self.strat_memory.add(np.stack([s[0] for s in strat]), np.array([s[1] for s in strat], np.float32), np.stack([s[2] for s in strat]))
                if self.algo == "dream" and q_data and self.shared_baseline:
                    self.q_memory[0].add(*(np.stack([d[i] for d in q_data]).astype(np.float32) for i in range(3)))
                elif self.algo == "dream" and q_data:
                    self.q_memory[p].add(*(np.stack([d[i] for d in q_data]).astype(np.float32) for i in range(3)))
                    xs, tgs, mks = self.q_memory[p].sample(min(self.q_memory[p].size, 200_000), self.rng)
                    self.q_nets[p].train()
                    _fit(self.q_nets[p], self.q_opts[p], xs, tgs, np.ones(len(xs), np.float32), self.q_steps, self.q_batch, self.device, masks=mks)
                    self.q_nets[p].eval()
                    self._q_tab[p] = None
                # advantage / regret net from scratch (linear CFR weights t)
                start = self._new_model()
                if self.warm_start and self.iteration > 1:
                    start.load_state_dict(self.nets[p].state_dict())
                self.nets[p] = _fit_from_buffer(start, self.adv_memory[p], self.adv_steps, self.adv_batch, self.lr, self.device,
                                                weight_scale=1.0 / t if self.normalized_weights else 1.0, grad_clip=self.grad_clip,
                                                legal_fn=self._legal_fn)
                self.iterates[p].append(self._cpu_state(self.nets[p]))
                self._sigma_tab[p] = None
            if self.algo == "dream" and self.shared_baseline and self.q_memory[0].size:  # once, after both players
                xs, tgs, mks = self.q_memory[0].sample(min(self.q_memory[0].size, 200_000), self.rng)
                self.q_nets[0].train()
                _fit(self.q_nets[0], self.q_opts[0], xs, tgs, np.ones(len(xs), np.float32), self.q_steps, self.q_batch, self.device, masks=mks)
                self.q_nets[0].eval()
                self._q_tab[0] = None
            self.stats = {"iteration": self.iteration, "seconds": time.perf_counter() - t0, "adv_samples": [len(m) for m in self.adv_memory],
                          "strat_samples": len(self.strat_memory), "nodes_touched": self.nodes_touched}
        return self

    # -- policies for evaluation ---------------------------------------------------------------
    def current_policy(self):
        """The strategy the next iteration plays (regret matching with the solver's own fallback)."""
        return NetPolicy(self.game, self.nets, "rm", self.device, argmax=self.rm_argmax)

    def policy_net(self):
        """DeepCFR / ESCHER average-strategy net fitted on the strategy memory."""
        if len(self.strat_memory) == 0:
            raise ValueError("no strategy samples")
        return _fit_from_buffer(self._new_model(policy=True), self.strat_memory, self.policy_steps, self.policy_batch, self.lr, self.device,
                                loss="policy", weight_scale=1.0 / max(self.iteration, 1) if self.normalized_weights else 1.0,
                                grad_clip=self.grad_clip)

    def average_policy(self):
        """Exact SD-CFR average: per infoset the reach-weighted (weight t * own reach) mixture of all
        iterates' regret-matching strategies (deepcfr/escher: the fitted policy net instead).  All
        iterates are evaluated at every infoset in one pass, then one tree walk per player carries the
        vector of the iterates' own reach probabilities."""
        if self.algo in ("deepcfr", "escher"):
            return NetPolicy(self.game, self.policy_net(), "softmax", self.device)
        T = len(self.iterates[0]) - 1
        weights = np.arange(1, T + 1, dtype=np.float64)
        net = self._new_model()
        num = np.zeros((len(self._info_keys), self.game.num_actions))
        den = np.zeros(len(self._info_keys))
        for p in range(2):
            rows = np.flatnonzero(self._info_player == p)
            sig = np.zeros((T, len(self._info_keys), self.game.num_actions))
            for t in range(1, T + 1):
                net.load_state_dict(self.iterates[p][t])
                out = self._forward(net, self._info_obs[rows])
                sig[t - 1, rows] = regret_matching_rows(out, self._info_legal[rows], self.rm_argmax)
            self._accumulate(self.game.new_initial_state(), p, np.ones(T), weights, sig, num, den)
        table = {k: num[i] / den[i] for i, k in enumerate(self._info_keys) if den[i] > 0}
        return TabularPolicy(self.game, table)

    def _accumulate(self, state, p, reach, weights, sig, num, den):
        """num[I] += sum_t w_t reach_t(I) sigma_t(I), den[I] += sum_t w_t reach_t(I) at p's infosets
        (``reach``: the iterates' own reach probabilities, shape (T,))."""
        if state.is_terminal():
            return
        if state.is_chance():
            for a, _ in state.chance_outcomes():
                self._accumulate(state.child(a), p, reach, weights, sig, num, den)
            return
        if state.current_player == p:
            i = self._info_index[state.info_key(p)]
            s = sig[:, i]  # (T, A)
            wr = weights * reach
            num[i] += wr @ s
            den[i] += wr.sum()
            for a in state.legal_actions():
                r = reach * s[:, a]
                if r.any():
                    self._accumulate(state.child(a), p, r, weights, sig, num, den)
        else:
            for a in state.legal_actions():
                self._accumulate(state.child(a), p, reach, weights, sig, num, den)

    # -- checkpoints ------------------------------------------------------------------------------
    def state_dict(self):
        state = {"iteration": self.iteration, "nodes_touched": self.nodes_touched, "nets": [self._cpu_state(n) for n in self.nets],
                 "iterates": [list(it) for it in self.iterates], "adv_memory": [m.state_dict() for m in self.adv_memory],
                 "adv_rng": [m.rng.bit_generator.state for m in self.adv_memory], "rng": self.rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state()}
        if self.algo in ("deepcfr", "escher"):
            state["strat_memory"], state["strat_rng"] = self.strat_memory.state_dict(), self.strat_memory.rng.bit_generator.state
        if self.algo == "dream":
            state["q_nets"] = [self._cpu_state(n) for n in self.q_nets]
            state["q_opts"] = [copy.deepcopy(o.state_dict()) for o in self.q_opts]  # an optimiser hands out its live tensors
            state["q_memory"] = [(m.x[: m.size].copy(), m.target[: m.size].copy(), m.mask[: m.size].copy(), m.pos, m.size) for m in self.q_memory]
        return state

    def load_state_dict(self, state):
        self.iteration, self.nodes_touched = int(state["iteration"]), int(state["nodes_touched"])
        for net, sd in zip(self.nets, state["nets"]):
            net.load_state_dict(sd)
        self.iterates = [list(it) for it in state["iterates"]]
        for m, sd, rs in zip(self.adv_memory, state["adv_memory"], state["adv_rng"]):
            m.load_state_dict(sd)
            m.rng.bit_generator.state = rs
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if "strat_memory" in state:
            self.strat_memory.load_state_dict(state["strat_memory"])
            self.strat_memory.rng.bit_generator.state = state["strat_rng"]
        if self.algo == "dream":
            if len(state["q_nets"]) != len(self.q_nets):
                raise ValueError(f"checkpoint has {len(state['q_nets'])} baseline net(s), this solver {len(self.q_nets)}: "
                                 "resume with the same --shared-baseline setting")
            for net, opt, sd, osd in zip(self.q_nets, self.q_opts, state["q_nets"], state["q_opts"]):
                net.load_state_dict(sd)
                _load_optimiser(opt, osd)
            for m, (x, tg, mk, pos, size) in zip(self.q_memory, state["q_memory"]):
                m.x[:size], m.target[:size], m.mask[:size], m.pos, m.size = x, tg, mk, pos, size
        self._sigma_tab, self._q_tab, self._v_tab = [None, None], [None, None], None
        return self

    def evaluate(self):
        """Exploitability of the current and the average strategy.  Evaluating draws from torch's generators (a
        fresh net's initialisation, the policy fit's minibatches): they are put back, so a run does not depend on
        how often it is evaluated."""
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        try:
            cur = exploitability(self.game, self.current_policy())[0]
            avg = exploitability(self.game, self.average_policy())[0]
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return {"current": cur, "average": avg}


def main(argv=None):
    from headsup.games import make_small_game

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--algo", default="deepcfr", choices=ALGOS)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--traversals", type=int, default=346, help="per iteration and player (SD-CFR Leduc: 346 ES; DREAM: 900 OS; ESCHER 1000)")
    p.add_argument("--adv-steps", type=int, default=3000)
    p.add_argument("--adv-batch", type=int, default=2048)
    p.add_argument("--adv-capacity", type=int, default=2_000_000, help="advantage reservoir per player (DREAM paper 2M, SD-CFR paper 1M)")
    p.add_argument("--strat-capacity", type=int, default=2_000_000)
    p.add_argument("--warm-start", action="store_true", help="advantage nets start from the player's previous net (SD-CFR paper's Leduc setup)")
    p.add_argument("--policy-steps", type=int, default=4000)
    p.add_argument("--policy-batch", type=int, default=2048)
    p.add_argument("--q-steps", type=int, default=1000, help="DREAM baseline / ESCHER value-net steps per iteration")
    p.add_argument("--q-batch", type=int, default=512)
    p.add_argument("--value-traversals", type=int, default=None, help="ESCHER: value trajectories per iteration (default --traversals)")
    p.add_argument("--epsilon", type=float, default=0.5, help="DREAM: traverser exploration")
    p.add_argument("--value-epsilon", type=float, default=0.01, help="ESCHER: exploration of the value trajectories")
    p.add_argument("--arch", default="mlp", choices=["mlp", "deepcfr", "deepcfr_dueling", "pokerrl", "pokerrl_nodueling", "pokerrl_nonorm", "pokerrl_nomask"],
                   help="networks: 3 x 64 MLP, the Deep CFR architecture (DREAM paper) or the SD-CFR authors' PokerRL nets")
    p.add_argument("--loss-weights", default="raw", choices=["raw", "normalized"], help="t, or t / t_latest (SD-CFR authors' code)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="gradient-norm clipping (Deep CFR paper 1; SD-CFR authors' code 10)")
    p.add_argument("--mean-regret", action="store_true", help="divide sampled regrets by the number of legal actions (SD-CFR authors' sampler)")
    p.add_argument("--shared-baseline", action="store_true",
                   help="DREAM: one baseline net (player 0's values) trained once per iteration on both traversers' data (authors' code)")
    p.add_argument("--bootstrap-chance", action="store_true",
                   help="DREAM: expected-SARSA targets bootstrap from the baseline after a sampled deal (authors' code)")
    p.add_argument("--masked-loss", action="store_true", help="advantage loss on legal actions only (what masked network outputs do)")
    p.add_argument("--rm-fallback", default="argmax", choices=["argmax", "uniform"],
                   help="regret matching without a positive advantage: the best action (Deep CFR / DREAM / ESCHER) or uniform")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--checkpoint", default=None, help="saved at every evaluation; an existing file is resumed from")
    args = p.parse_args(argv)
    game = make_small_game(args.game)
    solver = DeepSolver(game, args.algo, traversals=args.traversals, adv_steps=args.adv_steps, adv_batch=args.adv_batch,
                        policy_steps=args.policy_steps, policy_batch=args.policy_batch, q_steps=args.q_steps, q_batch=args.q_batch,
                        value_traversals=args.value_traversals, epsilon=args.epsilon, value_epsilon=args.value_epsilon,
                        device=args.device, seed=args.seed, rm_argmax=args.rm_fallback == "argmax", warm_start=args.warm_start,
                        adv_capacity=args.adv_capacity, strat_capacity=args.strat_capacity,
                        model_kwargs={"arch": args.arch} if args.arch != "mlp" else None,
                        normalized_weights=args.loss_weights == "normalized", grad_clip=args.grad_clip, mean_regret=args.mean_regret,
                        masked_loss=args.masked_loss, shared_baseline=args.shared_baseline, bootstrap_chance=args.bootstrap_chance)
    curve, elapsed = [], 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        solver.load_state_dict(saved["solver"])
        curve, elapsed = saved["curve"], saved["seconds"]
        print(f"resumed from {args.checkpoint} at iteration {solver.iteration}", flush=True)
    t0 = time.perf_counter() - elapsed
    for it in range(solver.iteration + 1, args.iterations + 1):
        solver.iterate()
        if it % args.eval_every == 0 or it == args.iterations:
            ev = solver.evaluate()
            curve.append({"iteration": it, **ev, "nodes_touched": solver.nodes_touched, "seconds": time.perf_counter() - t0})
            print(f"{args.game} {args.algo} it {it}: exploitability current {ev['current']:.4f} average {ev['average']:.4f}  "
                  f"nodes {solver.nodes_touched:.3g}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if args.checkpoint:  # atomic: a crash while writing leaves the previous checkpoint intact
                torch.save({"solver": solver.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}, args.checkpoint + ".tmp")
                os.replace(args.checkpoint + ".tmp", args.checkpoint)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"game": args.game, "algo": args.algo, "args": vars(args), "curve": curve}, f, indent=2)


if __name__ == "__main__":
    main()
