"""Deep CFR / SD-CFR / DREAM / ESCHER on the game protocol (Kuhn), against exact exploitability."""

import pytest

from headsup.algos.best_response import exploitability
from headsup.algos.deep import DeepSolver
from headsup.games import UniformPolicy, make_game


@pytest.mark.parametrize("algo", ["deepcfr", "sdcfr", "dream", "escher"])
def test_deep_algorithms_reduce_kuhn_exploitability(algo):
    g = make_game("kuhn")
    uniform = exploitability(g, UniformPolicy(g))[0]
    s = DeepSolver(g, algo, traversals=60, adv_steps=120, adv_batch=256, policy_steps=200, policy_batch=256,
                   q_steps=60, q_batch=128, seed=0)
    s.iterate(12)
    ev = s.evaluate()
    assert ev["average"] < 0.6 * uniform, ev  # uniform: 0.458; the average strategies reach ~0.05-0.25 here
    assert len(s.iterates[0]) == 13 and s.stats["iteration"] == 12
    # the exact SD-CFR average of the bank equals the reach-weighted mixture (sums to one everywhere)
    if algo in ("sdcfr", "dream"):
        pol = s.average_policy()
        for key, probs in pol.table.items():
            assert probs.sum() == pytest.approx(1.0, abs=1e-6)
