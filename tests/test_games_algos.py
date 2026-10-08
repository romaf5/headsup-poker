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


# exploitability of the average after 100 iterations (antes): vanilla 0.096, LCFR 0.034, CFR+ 0.013,
# DCFR 0.008, PCFR+ 0.020 (uniform: 2.37); regrets are applied once per iteration and every infoset's
# strategy sum is updated (in-place per-history updates had CFR+ at 0.035 after 1000 iterations)
@pytest.mark.parametrize("variant,bound", [("vanilla", 0.12), ("lcfr", 0.045), ("cfr+", 0.018), ("dcfr", 0.011), ("pcfr+", 0.026)])
def test_leduc_cfr_variants_converge(variant, bound):
    g = make_game("leduc")
    solver = CFR(g, variant).iterate(100)
    ex, _ = exploitability(g, solver.average_policy())
    assert ex < bound, ex


def test_leduc_value_from_a_long_cfr_plus_run():
    g = make_game("leduc")
    avg = CFR(g, "dcfr").iterate(150).average_policy()
    assert expected_value(g, avg) == pytest.approx(-0.0856, abs=0.002)  # the known value of Leduc for player 0
    assert exploitability(g, avg)[0] < 0.006


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


@pytest.mark.parametrize("name", ["kuhn", "leduc"])
def test_info_state_is_perfect_recall(name):
    """The network input determines the infoset up to the (strategically irrelevant) suits: two
    states with the same features must have the same ranks and the same betting history."""
    game = make_game(name)
    seen = {}

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        p = s.current_player
        board = None if s.board is None else game.rank_of(s.board)
        key = (p, game.rank_of(s.cards[p]), board, tuple(tuple(h) for h in s.history))
        assert seen.setdefault((p, s.info_state(p).tobytes()), key) == key
        for a in s.legal_actions():
            walk(s.child(a))

    walk(game.new_initial_state())


# measured (antes): Kuhn after 200 iterations 0.0012 / 0.0000, Leduc after 100 iterations 0.0086 / 0.0177 (DCFR 0.0078, PCFR+ 0.0198)
@pytest.mark.parametrize("variant,bound", [("dcfr+", 0.012), ("pdcfr+", 0.023)])
def test_discounted_plus_variants_converge(variant, bound):
    """DCFR+ / PDCFR+ (Xu et al. 2024, arXiv 2404.13891): discounted regrets floored at zero, predictive strategy."""
    kuhn = make_game("kuhn")
    avg = CFR(kuhn, variant).iterate(200).average_policy()
    assert exploitability(kuhn, avg)[0] < 0.003
    assert expected_value(kuhn, avg) == pytest.approx(-1 / 18, abs=0.003)
    leduc = make_game("leduc")
    assert exploitability(leduc, CFR(leduc, variant).iterate(100).average_policy())[0] < bound
    assert CFR(kuhn, "dcfr").gamma == 2.0 and CFR(kuhn, "dcfr+").gamma == 4.0 and CFR(kuhn, "pdcfr+").alpha == 2.3


def test_pdcfr_plus_predicts_with_the_discount_of_the_next_update():
    """An infoset updated in iteration t predicts with d_(t+1) from then on - also during the other player's walk
    later in the same iteration (alternating updates), where it used d_t."""
    s = CFR(make_game("kuhn"), "pdcfr+")
    legal = np.array([True, True, False])
    s.regret["x"] = np.array([2.0, 0.0, 0.0])
    s.last_regret["x"] = np.array([-1.0, 1.0, 0.0])
    s.iteration, s.last_update["x"] = 3, 3  # already updated in the current iteration
    d4 = 3**2.3 / (3**2.3 + 1)
    np.testing.assert_allclose(s._sigma("x", legal), np.array([2 * d4 - 1, 1.0, 0.0]) / (2 * d4))
    s.last_update["x"] = 2  # not yet updated in iteration 3
    d3 = 2**2.3 / (2**2.3 + 1)
    np.testing.assert_allclose(s._sigma("x", legal), np.array([2 * d3 - 1, 1.0, 0.0]) / (2 * d3))
