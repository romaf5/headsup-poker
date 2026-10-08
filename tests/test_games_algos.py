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


def test_oracle_cli_runs_all_requested_iterations(tmp_path):
    """Iterations beyond the last --eval point were never run (--iterations 300 with the default list ran 200)."""
    import json

    from headsup.algos.oracle import main

    main(["--game", "kuhn", "--algo", "escher", "--iterations", "30", "--eval", "1,10,20", "--trajectories", "5", "--json", str(tmp_path / "o.json")])
    curve = json.load(open(tmp_path / "o.json"))["curve"]
    assert [row["iteration"] for row in curve] == [1, 10, 20, 30]


def test_oracle_sampled_average_does_not_depend_on_the_update_players_exploration():
    """DREAM's update player samples epsilon-greedily from its CURRENT strategy, so how often a trajectory reaches an
    opponent infoset changes with that strategy; the sampled average must correct for it (1 / sampling reach). At
    player 1's infosets after a bet its own reach is 1: each trajectory must contribute mass 1 on average, whether
    player 0 always bets (0.75 uncorrected) or always checks (0.25)."""
    from headsup.algos.oracle import OracleSampler

    g = make_game("kuhn")
    for p0_bets in (True, False):
        s = OracleSampler(g, "dream", trajectories=1, epsilon=0.5, seed=0, average="sampled")
        root = g.new_initial_state()
        for c0 in range(3):
            for c1 in range(3):
                if c1 != c0:  # player 0's regrets: always bet (action 2) or always check (action 1)
                    r = np.zeros(3)
                    r[2 if p0_bets else 1] = 1.0
                    s.regret[root.child(c0).child(c1).info_key(0)] = r
        q_tab = {}
        s._values(g.new_initial_state(), q_tab)
        n = 20000
        for _ in range(n):
            s._dream(g.new_initial_state(), 0, 1.0, q_tab, {})
        mass = sum(v.sum() for k, v in s.strategy_sum.items() if k[0] == 1 and k[4][0] == (2,)) / n
        assert mass == pytest.approx(1.0, abs=0.06), (p0_bets, mass)


def test_small_game_solvers_refuse_holdem_presets():
    """`--game fhp` reached the solver with a GameConfig (AttributeError), or would enumerate the whole game."""
    from headsup.algos import deep, oracle

    for main in (deep.main, oracle.main):
        with pytest.raises(ValueError, match="small game"):
            main(["--game", "fhp", "--iterations", "1"])


@pytest.mark.parametrize("algo", ["escher", "dream"])
def test_oracle_importance_weighted_average_is_unbiased(algo):
    """--average own_is (how OpenSpiel's outcome sampling accumulates the average, which the ESCHER paper's tabular
    experiment uses): the update player's infosets get its own reach x sigma / sampling reach.  In expectation that is
    the exact reach-weighted strategy of every history of the infoset."""
    from headsup.algos.oracle import OracleSampler

    g = make_game("kuhn")
    rng = np.random.default_rng(0)
    s = OracleSampler(g, algo, trajectories=1, epsilon=0.5, seed=0, average="own_is")
    exact = OracleSampler(g, algo, trajectories=1, epsilon=0.5, seed=0, average="exact")
    stack = [g.new_initial_state()]
    while stack:  # random regrets: a mixed strategy at every infoset
        st = stack.pop()
        if st.is_terminal():
            continue
        if st.is_chance():
            stack.extend(st.child(a) for a, _ in st.chance_outcomes())
            continue
        key = st.info_key(st.current_player)
        if key not in s.regret:
            s.regret[key] = exact.regret[key] = np.where(st.legal_mask(), rng.random(g.num_actions) + 0.1, 0.0)
        stack.extend(st.child(a) for a in st.legal_actions())
    exact._accumulate_average(g.new_initial_state(), 0, 1.0)
    q_tab = {}
    s._values(g.new_initial_state(), q_tab)
    n = 60000
    for _ in range(n):
        if algo == "escher":
            s._escher(0, q_tab, {})
        else:
            s._dream(g.new_initial_state(), 0, 1.0, q_tab, {})
    assert set(s.strategy_sum) == set(exact.strategy_sum) and len(s.strategy_sum) == 6  # player 0's infosets only
    for key, want in exact.strategy_sum.items():
        np.testing.assert_allclose(s.strategy_sum[key] / n, want, atol=0.06)


def test_oracle_reports_the_pooled_regret_variance():
    """The ESCHER paper's variance pools all regret estimates of an iteration (legal actions); the within-infoset
    variance we reported is a different, smaller number (3.7 vs 5.1 for ESCHER, 120 vs 280 for DREAM on Leduc)."""
    from headsup.algos.oracle import OracleSampler

    s = OracleSampler(make_game("kuhn"), "dream", trajectories=200, epsilon=0.5, seed=0).iterate(3)
    assert len(s.variance) == len(s.variance_pooled) == 3 and all(v > 0 for v in s.variance_pooled)
    assert np.mean(s.variance_pooled) > np.mean(s.variance)  # pooling adds the spread between infosets


def test_mccfr_pruning_follows_the_pluribus_rule():
    """Pluribus (supplement): pruning is decided for the whole iteration, never applies on the last betting round or
    to actions that lead straight to a terminal node, and regrets have a floor.  Ours drew per node and had neither
    exemption (an action with a very negative regret could never be explored again at the end of the game) nor a floor."""
    g = make_game("leduc")
    root = g.new_initial_state().child(0).child(1)  # a deal; player 0 to act in the first round
    assert not root.is_chance() and root.current_player == 0
    key = root.info_key(0)
    raise_ = max(root.legal_actions())  # leads to another decision: prunable
    bet_node = root.child(raise_)
    fold = min(bet_node.legal_actions())  # facing the bet: folding ends the hand - never pruned
    assert bet_node.child(fold).is_terminal() and not root.child(raise_).is_terminal()
    m = MCCFR(g, "external", seed=0, prune_threshold=-1.0, prune_prob=1.0, prune_after=0, linear=False)
    m._get(m.regret, key)[raise_] = -1e6
    m._get(m.regret, bet_node.info_key(1))[fold] = -1e6
    m.iterate(300)
    assert m.regret[key][raise_] == -1e6  # pruned in every traversal: untouched
    assert m.regret[bet_node.info_key(1)][fold] != -1e6  # a terminal-leading action is always explored
    k = make_game("kuhn")  # one betting round = the last one: nothing is ever pruned
    mk = MCCFR(k, "external", seed=0, prune_threshold=-1.0, prune_prob=1.0, prune_after=0, linear=False)
    first = k.new_initial_state().child(0).child(1)
    a = max(first.legal_actions())
    mk._get(mk.regret, first.info_key(0))[a] = -1e6
    mk.iterate(200)
    assert mk.regret[first.info_key(0)][a] != -1e6
    floor = MCCFR(g, "external", seed=0, regret_floor=-0.25, linear=False).iterate(400)
    lowest = min(float(r.min()) for r in floor.regret.values())
    assert lowest >= -0.25 * 400 - 1e-9 and lowest < -50  # the floor on the average regret binds and holds
