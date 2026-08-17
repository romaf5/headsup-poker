"""Tabular CFR family on the game protocol - reference solvers and Pluribus-style blueprints.

``CFR``: full-width counterfactual regret minimisation with alternating updates and the usual
variants: ``vanilla`` (Zinkevich et al. 2007), ``lcfr`` (linear weights), ``cfr+`` (regret floor,
linear averaging; Tammelin 2014), ``dcfr`` (Brown & Sandholm 2019: alpha 1.5, beta 0, gamma 2),
``pcfr+`` (predictive RM+, quadratic averaging; Farina, Kroer & Sandholm 2021).
``MCCFR``: Monte-Carlo CFR (Lanctot et al. 2009) with external or outcome sampling, linear
weighting, and Pluribus' regret-based pruning (Brown & Sandholm 2019, Science: actions with very
negative regret are skipped on 95 % of the traversals) - the algorithm behind Pluribus' blueprint.

All keep ``regret[info_key]`` / ``strategy_sum[info_key]`` arrays over the game's actions and
expose ``average_policy()`` (a :class:`TabularPolicy`) for the exact exploitability check.
"""

import numpy as np

from headsup.algos.best_response import TabularPolicy


def regret_matching(regret, legal, prediction=None):
    r = regret + prediction if prediction is not None else regret
    pos = np.where(legal, np.maximum(r, 0.0), 0.0)
    total = pos.sum()
    if total > 0:
        return pos / total
    p = legal.astype(np.float64)
    return p / p.sum()


class _Tables:
    def __init__(self, game):
        self.game = game
        self.regret = {}
        self.strategy_sum = {}
        self.last_regret = {}
        self.iteration = 0

    def _get(self, table, key):
        v = table.get(key)
        if v is None:
            v = table[key] = np.zeros(self.game.num_actions)
        return v

    def average_policy(self):
        table = {}
        for key, s in self.strategy_sum.items():
            total = s.sum()
            if total > 0:
                table[key] = s / total
        return TabularPolicy(self.game, table)


class CFR(_Tables):
    def __init__(self, game, variant="cfr+", alpha=1.5, beta=0.0, gamma=2.0):
        super().__init__(game)
        assert variant in ("vanilla", "lcfr", "cfr+", "dcfr", "pcfr+")
        self.variant = variant
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._legal = {}

    def _sigma(self, key, legal):
        r = self._get(self.regret, key)
        pred = self.last_regret.get(key) if self.variant == "pcfr+" else None
        return regret_matching(r, legal, pred)

    def _walk(self, state, p, reach_p, reach_q):
        """Counterfactual value for player p; updates p's regrets and strategy sums."""
        if state.is_terminal():
            return state.returns()[p]
        if state.is_chance():
            return sum(prob * self._walk(state.child(a), p, reach_p, reach_q * prob) for a, prob in state.chance_outcomes())
        cur = state.current_player
        key = state.info_key(cur)
        legal = state.legal_mask()
        sigma = self._sigma(key, legal)
        if cur != p:
            v = 0.0
            for a in state.legal_actions():
                if sigma[a] > 0:
                    v += sigma[a] * self._walk(state.child(a), p, reach_p, reach_q * sigma[a])
            return v
        t = self.iteration
        values = np.zeros(self.game.num_actions)
        for a in state.legal_actions():
            values[a] = self._walk(state.child(a), p, reach_p * sigma[a], reach_q)
        v = float(sigma @ values)
        inst = np.where(legal, values - v, 0.0) * reach_q
        R = self._get(self.regret, key)
        S = self._get(self.strategy_sum, key)
        if self.variant == "vanilla":
            R += inst
            S += reach_p * sigma
        elif self.variant == "lcfr":
            R += t * inst
            S += t * reach_p * sigma
        elif self.variant == "dcfr":
            R += inst
            S += reach_p * sigma
        else:  # cfr+ / pcfr+
            np.maximum(R + inst, 0.0, out=R)
            self.last_regret[key] = inst
            S += (t if self.variant == "cfr+" else t * t) * reach_p * sigma
        return v

    def _discount(self):
        if self.variant != "dcfr":
            return
        t = self.iteration
        ta, tb = t**self.alpha, t**self.beta
        fp, fn, fs = ta / (ta + 1), tb / (tb + 1), (t / (t + 1)) ** self.gamma
        for R in self.regret.values():
            R *= np.where(R > 0, fp, fn)
        for S in self.strategy_sum.values():
            S *= fs

    def iterate(self, n=1):
        for _ in range(n):
            self.iteration += 1
            for p in range(2):
                self._walk(self.game.new_initial_state(), p, 1.0, 1.0)
            self._discount()
        return self


class MCCFR(_Tables):
    """External- or outcome-sampling MCCFR with linear weighting and optional regret-based pruning."""

    def __init__(self, game, sampling="external", seed=0, prune_threshold=None, prune_prob=0.95, prune_after=0, linear=True,
                 epsilon=0.6):
        super().__init__(game)
        assert sampling in ("external", "outcome")
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        # Pluribus: from iteration `prune_after` on, actions whose regret is below `prune_threshold` are
        # not traversed on a `prune_prob` fraction of the traversals (their regrets stay untouched)
        self.prune_threshold, self.prune_prob, self.prune_after = prune_threshold, prune_prob, prune_after
        self.linear = linear
        self.epsilon = epsilon

    def _sigma(self, key, legal):
        return regret_matching(self._get(self.regret, key), legal)

    def _external(self, state, p):
        if state.is_terminal():
            return state.returns()[p]
        if state.is_chance():
            return self._external(state.child(state.sample_chance(self.rng)), p)
        cur = state.current_player
        key = state.info_key(cur)
        legal = state.legal_mask()
        sigma = self._sigma(key, legal)
        t = self.iteration if self.linear else 1
        if cur != p:
            S = self._get(self.strategy_sum, key)
            S += t * sigma
            a = int(self.rng.choice(len(sigma), p=sigma))
            return self._external(state.child(a), p)
        R = self._get(self.regret, key)
        values = np.zeros(self.game.num_actions)
        explored = legal.copy()
        if self.prune_threshold is not None and self.iteration > self.prune_after and self.rng.random() < self.prune_prob:
            # threshold on the *average* regret per iteration (the sums grow like t^2 with linear weights)
            norm = self.iteration * (self.iteration + 1) / 2 if self.linear else self.iteration
            explored &= ~(R / norm < self.prune_threshold)
            if not explored.any():
                explored = legal.copy()
        for a in np.flatnonzero(explored):
            values[a] = self._external(state.child(a), p)
        v = float(np.where(explored, sigma, 0.0) @ values)  # Pluribus: the mean over the explored actions
        for a in np.flatnonzero(explored):
            R[a] += t * (values[a] - v)
        return v

    def _outcome(self, state, p, reach_p, reach_q, sample_prob):
        """Outcome sampling (Lanctot et al. 2009): one trajectory z sampled with the traverser's
        epsilon-greedy policy; returns (u_p(z) / q(z), own tail reach pi_p(h -> z), opponent tail
        reach pi_-p(h -> z)).  Regret estimate at the traverser's infoset with sampled action a:
        r(a') = W * (pi_p(h a' -> z) - pi_p(h -> z)), W = u_p(z) * pi_-p(z) / q(z)."""
        if state.is_terminal():
            return state.returns()[p] / sample_prob, 1.0, 1.0
        if state.is_chance():
            outcomes = state.chance_outcomes()
            i = int(self.rng.choice(len(outcomes), p=[pr for _, pr in outcomes]))
            a, pr = outcomes[i]
            return self._outcome(state.child(a), p, reach_p, reach_q * pr, sample_prob * pr)
        cur = state.current_player
        key = state.info_key(cur)
        legal = state.legal_mask()
        sigma = self._sigma(key, legal)
        t = self.iteration if self.linear else 1
        if cur == p:
            explore = legal / legal.sum()
            q = self.epsilon * explore + (1 - self.epsilon) * sigma
        else:
            q = sigma
        a = int(self.rng.choice(len(q), p=q / q.sum()))
        child = state.child(a)
        if cur == p:
            u, tail_p, tail_q = self._outcome(child, p, reach_p * sigma[a], reach_q, sample_prob * q[a])
            W = u * reach_q * tail_q
            R = self._get(self.regret, key)
            for b in np.flatnonzero(legal):
                R[b] += t * W * ((tail_p if b == a else 0.0) - sigma[a] * tail_p)
            return u, tail_p * sigma[a], tail_q
        u, tail_p, tail_q = self._outcome(child, p, reach_p, reach_q * sigma[a], sample_prob * q[a])
        S = self._get(self.strategy_sum, key)
        S += t * (reach_p / sample_prob) * sigma  # stochastically weighted averaging
        return u, tail_p, tail_q * sigma[a]

    def iterate(self, n=1):
        for _ in range(n):
            self.iteration += 1
            for p in range(2):
                if self.sampling == "external":
                    self._external(self.game.new_initial_state(), p)
                else:
                    self._outcome(self.game.new_initial_state(), p, 1.0, 1.0, 1.0)
        return self
