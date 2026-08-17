"""Game protocol (Kuhn, Leduc), exact best response, tabular CFR / MCCFR reference solvers."""

import numpy as np
import pytest

from headsup.algos.best_response import best_response, expected_value, exploitability
from headsup.algos.tabular import CFR, MCCFR
from headsup.games import UniformPolicy, make_game


def _sizes(game):
    infosets, terminals = set(), 0

    def walk(s):
        nonlocal terminals
        if s.is_terminal():
            terminals += 1
            return
        if s.is_chance():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        infosets.add(s.info_key(s.current_player))
        for a in s.legal_actions():
            walk(s.child(a))

    walk(game.new_initial_state())
    return len(infosets), terminals


def test_kuhn_and_leduc_have_the_known_sizes():
    assert _sizes(make_game("kuhn")) == (12, 30)
    assert _sizes(make_game("leduc"))[0] == 936
    g = make_game("leduc")
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = g.playout([UniformPolicy(g)] * 2, rng).returns()
        assert r[0] == -r[1]
        assert abs(r[0]) <= 13


def test_kuhn_cfr_plus_reaches_the_equilibrium_value():
    g = make_game("kuhn")
    solver = CFR(g, "cfr+").iterate(500)
    avg = solver.average_policy()
    ex, br = exploitability(g, avg)
    assert ex < 0.01, ex
    assert expected_value(g, avg) == pytest.approx(-1 / 18, abs=0.005)  # the value of Kuhn poker
    # best response beats the uniform policy by a lot
    assert best_response(g, UniformPolicy(g), 0)[0] > 0.4


@pytest.mark.parametrize("variant", ["vanilla", "lcfr", "cfr+", "dcfr", "pcfr+"])
def test_leduc_cfr_variants_converge(variant):
    g = make_game("leduc")
    solver = CFR(g, variant).iterate(30)
    ex, _ = exploitability(g, solver.average_policy())
    assert ex < 0.6, ex  # uniform: 2.37; the accelerated variants are ~0.2 here
    if variant in ("cfr+", "dcfr"):
        assert ex < 0.3, ex


def test_leduc_value_from_a_long_cfr_plus_run():
    g = make_game("leduc")
    avg = CFR(g, "dcfr").iterate(150).average_policy()
    assert expected_value(g, avg) == pytest.approx(-0.0856, abs=0.015)  # the known value of Leduc for player 0
    assert exploitability(g, avg)[0] < 0.08


def test_mccfr_external_and_outcome_sampling_converge():
    k = make_game("kuhn")
    ext = MCCFR(k, "external", seed=0).iterate(8000)
    assert exploitability(k, ext.average_policy())[0] < 0.03
    out = MCCFR(k, "outcome", seed=0).iterate(15000)
    assert exploitability(k, out.average_policy())[0] < 0.06
    g = make_game("leduc")
    ext = MCCFR(g, "external", seed=1).iterate(1500)
    assert exploitability(g, ext.average_policy())[0] < 1.0
    pruned = MCCFR(g, "external", seed=1, prune_threshold=-1.0, prune_after=500).iterate(1500)
    assert exploitability(g, pruned.average_policy())[0] < 1.1
