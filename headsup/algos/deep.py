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
* ``escher`` (McAleer et al. 2022): the update player samples uniformly (a fixed sampling policy),
  the opponent from sigma; a history value net q(h, a) (both players' infostates -> player 0's
  value per action) fitted on Monte-Carlo returns of self-play trajectories; the regret estimate
  is q_i(h, a) - sum_a sigma_i(s, a) q_i(h, a) with no importance weights; cumulative regret buffer
  -> regret net from scratch each iteration; average policy net from the visited infosets.

Networks come from ``game.make_model()``; ``policy_*`` helpers wrap them for the exact
:func:`headsup.algos.best_response.exploitability` check.  Small games only (Python traversal);
the hold'em pipeline in headsup.deepcfr keeps its C++ kernels.
"""

import argparse
import json
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


def _fit(model, opt, xs, targets, weights, steps, batch, rng, device, masks=None, grad_clip=1.0):
    """Weighted MSE regression steps on numpy arrays; ``masks`` restricts the loss to some outputs."""
    xs_t = torch.as_tensor(xs, device=device)
    tg_t = torch.as_tensor(targets, device=device)
    w_t = torch.as_tensor(weights, device=device)
    m_t = torch.as_tensor(masks, device=device) if masks is not None else None
    n = len(xs)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, batch), device=device)
        pred = model(xs_t[idx])
        err = (pred - tg_t[idx]).pow(2)
        if m_t is not None:
            err = err * m_t[idx]
        loss = (w_t[idx][:, None] * err).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=False)
        opt.step()
    return float(loss.item())


def _fit_from_buffer(model, buffer, steps, batch, lr, device, loss="mse", weight_power=1.0):
    """DeepCFR-style fit on a ReservoirBuffer: targets weighted by t^power; ``loss`` mse (advantages)
    or 'policy' (softmax(logits) vs stored probabilities)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for obs, t, target in buffer.prefetch(batch, steps):
        pred = model(obs)
        if loss == "policy":
            pred = torch.softmax(pred, dim=-1)
        l = ((t.pow(weight_power))[:, None] * (pred - target).pow(2)).mean()
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        opt.step()
    model.eval()
    return model


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

    def __init__(self, game, nets, mode="rm", device="cpu"):
        self.game, self.nets, self.mode, self.device = game, nets, mode, device
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
                probs = regret_matching_np(out, legal)
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
                 q_capacity=200_000, value_traversals=None, epsilon=0.5, lr=1e-3, device="cpu", seed=0, model_kwargs=None,
                 rm_argmax=False):
        assert algo in ALGOS
        self.game, self.algo = game, algo
        self.traversals = traversals
        self.value_traversals = value_traversals or traversals
        self.adv_steps, self.adv_batch = adv_steps, adv_batch
        self.policy_steps, self.policy_batch = policy_steps, policy_batch
        self.q_steps, self.q_batch = q_steps, q_batch
        self.epsilon, self.lr = epsilon, lr
        self.rm_argmax = rm_argmax
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.model_kwargs = model_kwargs or {}
        A, D = game.num_actions, game.obs_dim
        self.nets = [self._new_model() for _ in range(2)]
        self.iterates = [[self._cpu_state(n)] for n in self.nets]
        self.adv_memory = [ReservoirBuffer(adv_capacity, self.device, obs_dim=D, target_dim=A, seed=seed + i, int_dim=0) for i in range(2)]
        self.strat_memory = ReservoirBuffer(strat_capacity, self.device, obs_dim=D, target_dim=A, seed=seed + 2, int_dim=0)
        if algo == "dream":  # baseline Q_i(s*(h), a) per player, expected-SARSA targets
            self.q_nets = [self._new_model(2 * D) for _ in range(2)]
            self.q_opts = [torch.optim.Adam(n.parameters(), lr=lr) for n in self.q_nets]
            self.q_memory = [CircularBuffer(q_capacity, 2 * D, A) for _ in range(2)]
        if algo == "escher":  # history value q(h, a): player 0's expected return per action
            self.v_net = self._new_model(2 * D)
            self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)
            self.v_memory = CircularBuffer(q_capacity, 2 * D, A)
        self.iteration = 0
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
    def _new_model(self, in_dim=None):
        m = self.game.make_model(**self.model_kwargs) if in_dim is None else self.game.make_model(in_dim=in_dim, **self.model_kwargs)
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
        adv.append((state.info_state(p), t, np.where(legal, values - v, 0.0)))
        return v

    # -- outcome sampling with baselines (DREAM) --------------------------------------------------
    def _q(self, p, state):
        if self._q_tab[p] is None:
            self._refresh_q(p)
        return self._q_tab[p][state.history_key()]

    def _os_dream(self, state, p, t, own_sample_reach, adv, q_data):
        """Returns the baseline-corrected sampled value of ``state`` for player p (DREAM eq. 6-7)."""
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
        row_target[a] = target
        q_data.append((hist, row_target, row_mask))
        return v

    # -- ESCHER -------------------------------------------------------------------------------
    def _v(self, state):
        if self._v_tab is None:
            self._refresh_v()
        return self._v_tab[state.history_key()]

    def _escher_value_data(self, data):
        """One self-play trajectory under the current policies; (h, a, return of player 0) rows."""
        state = self.game.new_initial_state()
        rows = []
        while not state.is_terminal():
            if state.is_chance():
                state = state.child(state.sample_chance(self.rng))
                continue
            sigma = self._sigma(state.current_player, state)
            a = int(self.rng.choice(len(sigma), p=sigma))
            rows.append((self._history(state), a))
            state = state.child(a)
        u0 = state.returns()[0]
        for hist, a in rows:
            tgt = np.zeros(self.game.num_actions, np.float32)
            msk = np.zeros(self.game.num_actions, np.float32)
            tgt[a], msk[a] = u0, 1.0
            data.append((hist, tgt, msk))

    def _escher_regrets(self, p, t, adv, strat):
        """One trajectory: update player p samples uniformly, the opponent from sigma; regrets from q."""
        state = self.game.new_initial_state()
        while not state.is_terminal():
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
                strat.append((state.info_state(p), t, sigma))
                a = int(self.rng.choice(np.flatnonzero(legal)))
            else:
                a = int(self.rng.choice(len(sigma), p=sigma))
            state = state.child(a)

    # -- one iteration --------------------------------------------------------------------------
    def iterate(self, n=1):
        for _ in range(n):
            self.iteration += 1
            t = float(self.iteration)
            t0 = time.perf_counter()
            if self.algo == "escher":  # 1. retrain the history value net on fresh self-play data
                data = []
                for _ in range(self.value_traversals):
                    self._escher_value_data(data)
                x, tg, mk = (np.stack([d[i] for d in data]).astype(np.float32) for i in range(3))
                self.v_memory.add(x, tg, mk)
                self.v_net.train()
                xs, tgs, mks = self.v_memory.sample(min(self.v_memory.size, 200_000), self.rng)
                _fit(self.v_net, self.v_opt, xs, tgs, np.ones(len(xs), np.float32), self.q_steps, self.q_batch, self.rng, self.device, masks=mks)
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
                if strat:
                    self.strat_memory.add(np.stack([s[0] for s in strat]), np.array([s[1] for s in strat], np.float32), np.stack([s[2] for s in strat]))
                if self.algo == "dream" and q_data:
                    self.q_memory[p].add(*(np.stack([d[i] for d in q_data]).astype(np.float32) for i in range(3)))
                    xs, tgs, mks = self.q_memory[p].sample(min(self.q_memory[p].size, 200_000), self.rng)
                    self.q_nets[p].train()
                    _fit(self.q_nets[p], self.q_opts[p], xs, tgs, np.ones(len(xs), np.float32), self.q_steps, self.q_batch, self.rng, self.device, masks=mks)
                    self.q_nets[p].eval()
                    self._q_tab[p] = None
                # advantage / regret net from scratch (linear CFR weights t)
                self.nets[p] = _fit_from_buffer(self._new_model(), self.adv_memory[p], self.adv_steps, self.adv_batch, self.lr, self.device)
                self.iterates[p].append(self._cpu_state(self.nets[p]))
                self._sigma_tab[p] = None
            self.stats = {"iteration": self.iteration, "seconds": time.perf_counter() - t0, "adv_samples": [len(m) for m in self.adv_memory], "strat_samples": len(self.strat_memory)}
        return self

    # -- policies for evaluation ---------------------------------------------------------------
    def current_policy(self):
        return NetPolicy(self.game, self.nets, "rm", self.device)

    def policy_net(self):
        """DeepCFR / ESCHER average-strategy net fitted on the strategy memory."""
        if len(self.strat_memory) == 0:
            raise ValueError("no strategy samples")
        return _fit_from_buffer(self._new_model(), self.strat_memory, self.policy_steps, self.policy_batch, self.lr, self.device, loss="policy")

    def average_policy(self):
        """Exact SD-CFR average: per infoset the reach-weighted (weight t * own reach) mixture of all
        iterates' regret-matching strategies (deepcfr/escher: the fitted policy net instead)."""
        if self.algo in ("deepcfr", "escher"):
            return NetPolicy(self.game, self.policy_net(), "softmax", self.device)
        num, den = {}, {}
        net = self._new_model()
        for t in range(1, len(self.iterates[0])):
            for p in range(2):
                net.load_state_dict(self.iterates[p][t])
                out = self._forward(net, self._info_obs)
                tab = {k: regret_matching_np(out[i], self._info_legal[i], self.rm_argmax) for i, k in enumerate(self._info_keys) if self._info_player[i] == p}
                self._accumulate(self.game.new_initial_state(), p, float(t), 1.0, lambda st, tab=tab: tab[st.info_key(st.current_player)], num, den)
        table = {k: num[k] / den[k] for k in num if den[k] > 0}
        return TabularPolicy(self.game, table)

    def _accumulate(self, state, p, w, own_reach, sigma_t, num, den):
        if state.is_terminal():
            return
        if state.is_chance():
            for a, _ in state.chance_outcomes():
                self._accumulate(state.child(a), p, w, own_reach, sigma_t, num, den)
            return
        cur = state.current_player
        if cur == p:
            key = state.info_key(p)
            sig = sigma_t(state)
            num[key] = num.get(key, 0.0) + w * own_reach * sig
            den[key] = den.get(key, 0.0) + w * own_reach
            for a in state.legal_actions():
                if sig[a] > 0:
                    self._accumulate(state.child(a), p, w, own_reach * sig[a], sigma_t, num, den)
        else:
            for a in state.legal_actions():
                self._accumulate(state.child(a), p, w, own_reach, sigma_t, num, den)

    def evaluate(self):
        cur = exploitability(self.game, self.current_policy())[0]
        avg = exploitability(self.game, self.average_policy())[0]
        return {"current": cur, "average": avg}


def main(argv=None):
    from headsup.games import make_game

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--algo", default="deepcfr", choices=ALGOS)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--traversals", type=int, default=346, help="per iteration and player (SD-CFR Leduc: 346 ES; DREAM: 900 OS; ESCHER 1000)")
    p.add_argument("--adv-steps", type=int, default=3000)
    p.add_argument("--adv-batch", type=int, default=2048)
    p.add_argument("--policy-steps", type=int, default=4000)
    p.add_argument("--q-steps", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    game = make_game(args.game)
    solver = DeepSolver(game, args.algo, traversals=args.traversals, adv_steps=args.adv_steps, adv_batch=args.adv_batch,
                        policy_steps=args.policy_steps, q_steps=args.q_steps, epsilon=args.epsilon, device=args.device, seed=args.seed)
    curve = []
    t0 = time.perf_counter()
    for it in range(1, args.iterations + 1):
        solver.iterate()
        if it % args.eval_every == 0 or it == args.iterations:
            ev = solver.evaluate()
            curve.append({"iteration": it, **ev, "seconds": time.perf_counter() - t0})
            print(f"{args.game} {args.algo} it {it}: exploitability current {ev['current']:.4f} average {ev['average']:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"game": args.game, "algo": args.algo, "args": vars(args), "curve": curve}, f, indent=2)


if __name__ == "__main__":
    main()
