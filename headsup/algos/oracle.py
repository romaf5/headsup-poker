"""Tabular regret estimators with oracle history values: the ESCHER paper's Leduc experiment.

McAleer et al. (2023, §3 and Fig. 3) compare, on Leduc, tabular versions of three sampled regret
estimators, each with access to the exact history values of the current strategies:

* ``escher`` (their Algorithm 1): the update player samples actions from a *fixed* policy (uniform),
  the opponent from its current strategy; at every update-player state of the trajectory, for every
  action, the regret estimate is q_i(h, a) - v_i(h) - no importance weights.
* ``dream``: outcome sampling (update player on eps-uniform + (1 - eps) sigma), baseline-corrected
  sampled values with the oracle q as baseline (DREAM / VR-MCCFR), divided by the update player's
  sampling reach.
* ``os``: outcome-sampling MCCFR (Lanctot et al. 2009) - :class:`headsup.algos.tabular.MCCFR`.

An iteration = ``trajectories`` sampled trajectories per player with the strategy held fixed (the
paper does not state its count; 500 reproduces its OS-MCCFR curve and is its code's default), then
regret matching on the accumulated (unweighted) regrets.  The average strategy is exact: each
iteration adds pi_i(I) sigma(I) at every infoset (full tree walk).  Exploitability is reported as the
paper plots it: NashConv (the *sum* of both players' best-response gains, in antes).

    python -m headsup.algos.oracle --algo escher --iterations 1000
"""

import argparse
import json
import time

import numpy as np

from headsup.algos.best_response import TabularPolicy, exploitability
from headsup.algos.tabular import MCCFR, regret_matching


class OracleSampler:
    def __init__(self, game, algo="escher", trajectories=500, epsilon=0.5, seed=0):
        assert algo in ("escher", "dream")
        self.game, self.algo, self.k, self.epsilon = game, algo, trajectories, epsilon
        self.rng = np.random.default_rng(seed)
        self.regret, self.strategy_sum = {}, {}
        self.iteration = 0
        self.variance = []  # per iteration: mean over (infoset, action) of the estimator's sample variance

    # -- strategies -------------------------------------------------------------------------------
    def sigma(self, state):
        key = state.info_key(state.current_player)
        r = self.regret.get(key)
        legal = state.legal_mask()
        return regret_matching(r if r is not None else np.zeros(self.game.num_actions), legal)

    def _values(self, state, table):
        """Player 0's value of every history under the current strategies (the oracle); fills ``table``."""
        if state.is_terminal():
            return state.returns()[0]
        if state.is_chance():
            v = sum(p * self._values(state.child(a), table) for a, p in state.chance_outcomes())
        else:
            sig = self.sigma(state)
            q = np.zeros(self.game.num_actions)
            for a in state.legal_actions():
                q[a] = self._values(state.child(a), table)
            table[state.history_key()] = q
            v = float(sig @ q)
        return v

    # -- estimators -------------------------------------------------------------------------------
    def _escher(self, p, q_tab, acc):
        state = self.game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance():
                state = state.child(state.sample_chance(self.rng))
                continue
            legal = state.legal_mask()
            sig = self.sigma(state)
            if state.current_player == p:
                q = q_tab[state.history_key()] * (1.0 if p == 0 else -1.0)
                r = np.where(legal, q - sig @ np.where(legal, q, 0.0), 0.0)
                acc.setdefault(state.info_key(p), []).append(r)
                a = int(self.rng.choice(np.flatnonzero(legal)))  # the fixed (uniform) sampling policy
            else:
                a = int(self.rng.choice(len(sig), p=sig))
            state = state.child(a)

    def _dream(self, state, p, own_reach, q_tab, acc):
        """Baseline-corrected sampled value of ``state`` for p; appends the regret estimates."""
        if state.is_terminal():
            return state.returns()[p]
        if state.is_chance():
            return self._dream(state.child(state.sample_chance(self.rng)), p, own_reach, q_tab, acc)
        legal = state.legal_mask()
        sig = self.sigma(state)
        xi = self.epsilon * legal / legal.sum() + (1 - self.epsilon) * sig if state.current_player == p else sig
        a = int(self.rng.choice(len(xi), p=xi / xi.sum()))
        b = np.where(legal, q_tab[state.history_key()] * (1.0 if p == 0 else -1.0), 0.0)
        child_reach = own_reach * (xi[a] if state.current_player == p else 1.0)
        va = b.copy()
        va[a] = b[a] + (self._dream(state.child(a), p, child_reach, q_tab, acc) - b[a]) / xi[a]
        v = float(sig @ va)
        if state.current_player == p:
            acc.setdefault(state.info_key(p), []).append(np.where(legal, va - v, 0.0) / own_reach)
        return v

    # -- iterations -------------------------------------------------------------------------------
    def _accumulate_average(self, state, p, reach):
        if state.is_terminal() or reach == 0.0:
            return
        if state.is_chance():
            for a, _ in state.chance_outcomes():
                self._accumulate_average(state.child(a), p, reach)
            return
        sig = self.sigma(state)
        if state.current_player == p:
            key = state.info_key(p)
            s = self.strategy_sum.get(key)
            if s is None:
                s = self.strategy_sum[key] = np.zeros(self.game.num_actions)
            s += reach * sig
            for a in state.legal_actions():
                self._accumulate_average(state.child(a), p, reach * sig[a])
        else:
            for a in state.legal_actions():
                self._accumulate_average(state.child(a), p, reach)

    def iterate(self, n=1):
        for _ in range(n):
            self.iteration += 1
            for p in (0, 1):
                self._accumulate_average(self.game.new_initial_state(), p, 1.0)
            var = []
            for p in (0, 1):  # alternating: player 1 sees player 0's updated strategy (and fresh oracle values)
                q_tab = {}
                self._values(self.game.new_initial_state(), q_tab)
                acc = {}
                for _ in range(self.k):
                    if self.algo == "escher":
                        self._escher(p, q_tab, acc)
                    else:
                        self._dream(self.game.new_initial_state(), p, 1.0, q_tab, acc)
                for key, rs in acc.items():
                    rs = np.asarray(rs)
                    R = self.regret.get(key)
                    if R is None:
                        R = self.regret[key] = np.zeros(self.game.num_actions)
                    R += rs.sum(axis=0) / self.k  # an unbiased (up to the fixed per-infoset scale) iteration regret
                    if len(rs) > 1:
                        var.append(rs.var(axis=0, ddof=1).mean())
            self.variance.append(float(np.mean(var)) if var else float("nan"))
        return self

    def average_policy(self):
        table = {k: s / s.sum() for k, s in self.strategy_sum.items() if s.sum() > 0}
        return TabularPolicy(self.game, table)


def main(argv=None):
    from headsup.games import make_game

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--algo", default="escher", choices=["escher", "dream", "os"])
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--trajectories", type=int, default=500, help="sampled trajectories per player and iteration")
    p.add_argument("--epsilon", type=float, default=0.5, help="DREAM / OS: exploration of the update player")
    p.add_argument("--eval", default="1,2,5,10,20,50,100,200,500,1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    game = make_game(args.game)
    points = sorted(int(x) for x in args.eval.split(",") if int(x) <= args.iterations)
    if args.algo == "os":
        solver = MCCFR(game, "outcome", seed=args.seed, linear=False, epsilon=args.epsilon)
        step = lambda n: solver.iterate(n * args.trajectories)  # MCCFR: one trajectory per player per call
    else:
        solver = OracleSampler(game, args.algo, args.trajectories, args.epsilon, args.seed)
        step = solver.iterate
    curve, done, t0 = [], 0, time.perf_counter()
    for it in points:
        step(it - done)
        done = it
        nash_conv = 2 * exploitability(game, solver.average_policy())[0]
        row = {"iteration": it, "nash_conv": nash_conv, "seconds": time.perf_counter() - t0}
        if args.algo != "os":
            row["variance"] = float(np.nanmean(solver.variance))
        curve.append(row)
        print(f"{args.game} tabular {args.algo} it {it}: NashConv {nash_conv:.4f}" + (f"  regret-estimator variance {row['variance']:.3g}" if "variance" in row else "")
              + f"  ({row['seconds']:.0f}s)", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"game": args.game, "algo": args.algo, "args": vars(args), "curve": curve}, f, indent=2)


if __name__ == "__main__":
    main()
