"""Tabular MCCFR-P blueprint (headsup.blueprint / headsup_cpp.TabularBlueprint)."""

import numpy as np
import pytest

from headsup import native
from headsup.game import DEFAULT_GAME, FHP

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")


def test_preflop_classes_and_ehs():
    from headsup.blueprint import TabularBlueprint

    bp = TabularBlueprint(FHP, buckets=20, samples=100)
    cpp = bp.cpp
    mod = native.module()
    # 169 lossless pre-flop classes: pairs 0..12, suited 13..90, offsuit 91..168, symmetric in the cards
    keys = set()
    for a in range(52):
        for b in range(a + 1, 52):
            k = cpp.bucket(0, a, b, [])
            assert k == cpp.bucket(0, b, a, []) and 0 <= k < 169
            keys.add(k)
    assert len(keys) == 169
    assert cpp.bucket(0, 12, 25, []) == 12  # AA
    assert cpp.bucket(0, 12, 11, []) == 13 + 12 * 11 // 2 + 11  # AKs
    assert cpp.bucket(0, 12, 24, []) == 91 + 12 * 11 // 2 + 11  # AKo
    # river EHS is exact: P(win) + P(tie)/2 over the 990 opponent hands = the BoardTable equity
    bp = TabularBlueprint(DEFAULT_GAME, buckets=20, samples=100)
    board = [4, 18, 33, 47, 8]
    t = mod.BoardTable(board, 5)
    for c0, c1 in ((21, 34), (0, 1), (9, 22)):
        h = mod.combo_index(min(c0, c1), max(c0, c1))
        assert bp.cpp.ehs(c0, c1, board) == pytest.approx(float(np.array(t.equity)[h]), abs=1e-6)
    # Monte-Carlo EHS on the flop is unbiased-ish: strong hands high, weak hands low
    assert bp.cpp.ehs(12, 25, [4, 18, 33], 1) > 0.7 > 0.4 > bp.cpp.ehs(0, 14, [4, 18, 33], 1)  # AA vs 23o on 6s 7h 9d


def test_bucket_edges_are_monotone_and_cover_the_range():
    from headsup.blueprint import TabularBlueprint

    bp = TabularBlueprint(DEFAULT_GAME, buckets=50, samples=100).fit_abstraction(5000, 0, 4)
    for r in (1, 2, 3):
        e = np.array(bp.cpp.edges[r])
        assert len(e) == 49 and np.all(np.diff(e) >= 0) and 0.0 < e[0] and e[-1] < 1.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        cards = rng.choice(52, 7, replace=False)
        assert 0 <= bp.cpp.bucket(3, int(cards[0]), int(cards[1]), [int(c) for c in cards[2:]]) < 50


def test_training_reduces_fhp_exploitability_and_the_player_round_trips(tmp_path):
    from headsup.algos.holdem_br import HoldemBestResponse
    from headsup.blueprint import TabularBlueprint, TabularPlayer
    from headsup.env import make_vec_env, play_hands
    from headsup.players import make_player

    bp = TabularBlueprint(FHP, buckets=30, samples=100).fit_abstraction(5000, 0, 4)
    bp.configure(strategy_interval=5, lcfr_iterations=20000, discount_interval=1000, prune_after=15000)
    boards = [tuple(int(c) for c in np.random.default_rng(1).permutation(52)[:5]) for _ in range(6)]
    br = HoldemBestResponse(TabularPlayer(bp, seed=0), FHP, boards=len(boards))
    from headsup.engine import HeadsUpPoker
    from headsup.lbr import NUM_COMBOS

    root = HeadsUpPoker(rng=np.random.default_rng(0), game=FHP)
    root.reset()
    r = np.ones(NUM_COMBOS)
    early = br.evaluate_from(root, [r, r], boards)["exploitability_mbb"]  # untrained: uniform strategy
    bp.run(30000, 1, 4)
    br.player, br.probs_cache = TabularPlayer(bp, seed=0), {}
    late = br.evaluate_from(root, [r, r], boards)["exploitability_mbb"]
    assert late < 0.5 * early, (early, late)
    # save / load / spec round trip and play
    bp.save(tmp_path / "bp.pt")
    p = make_player(f"tab:{tmp_path / 'bp.pt'}", seed=0)
    assert p.game == FHP and p.bp.iterations == 30000
    q = make_player(f"tab:{tmp_path / 'bp.pt'}@current", seed=0)
    assert q.current
    env = make_vec_env(4, "call", seed=0, game=FHP)
    rewards = play_hands(env, p, 8)
    assert len(rewards) == 8
    # hand-substituted rows of one public state come back as one strategy per hand
    from headsup.lbr import substitute_hands

    e = HeadsUpPoker(rng=np.random.default_rng(3), game=FHP)
    e.reset()
    e.step(1)
    e.step(1)  # flop
    rows = substitute_hands(e.observation(1))
    probs = p.probs(rows)
    assert probs.shape == (NUM_COMBOS, FHP.num_actions)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)
    assert np.all(probs[:, 0] == 0)  # nothing to call: never fold


def test_table_abstraction_features_and_round_trip(tmp_path):
    from headsup.blueprint import TabularBlueprint, TabularPlayer

    mod = native.module()
    bp = TabularBlueprint(DEFAULT_GAME, buckets=30, mode="table", completions=40)
    board = [4, 18, 33, 47, 8]
    # river: the features are the exact equities (std 0), matching the BoardTable
    f = np.array(bp.cpp.features(3, board))
    eq = np.array(mod.BoardTable(board, 5).equity)
    ok = eq >= 0
    np.testing.assert_allclose(f[ok, 0], eq[ok], atol=1e-6)
    assert np.all(f[ok, 1] == 0) and np.all(f[~ok, 0] < 0)
    # turn: mean over all rivers, std > 0 for drawing hands; the set of tens (Th Td) is strong
    f = np.array(bp.cpp.features(2, board[:4]))
    h = mod.combo_index(21, 34)
    assert f[h, 0] > 0.75 and f[:, 1][ok].max() > 0.05
    bp.fit_abstraction(20 * 1326, 0, 4)  # 20 random boards per round
    assert [len(c) // 2 for c in bp.cpp.centroids] == [0, 30, 30, 30]
    b = np.array(bp.cpp.strategy_for_hands(0, np.array([[12, 25], [0, 14]], dtype=np.int32), [], 0))  # preflop still works
    assert b.shape == (2, 4)
    bp.configure(strategy_interval=5)
    bp.run(3000, 1, 4)
    assert bp.cpp.cache_sizes[1] > 0 and bp.cpp.cache_sizes[2] > 0  # per-board caches filled by the traversals
    bp.save(tmp_path / "t.pt")
    other = TabularBlueprint.load(tmp_path / "t.pt")
    assert other.mode == "table" and other.cpp.table_mode and [len(c) for c in other.cpp.centroids] == [len(c) for c in bp.cpp.centroids]
    p = TabularPlayer(other, seed=0)
    from headsup.engine import HeadsUpPoker
    from headsup.lbr import NUM_COMBOS, substitute_hands

    e = HeadsUpPoker(rng=np.random.default_rng(3))
    e.reset()
    e.step(1)
    e.step(1)
    probs = p.probs(substitute_hands(e.observation(1)))
    assert probs.shape == (NUM_COMBOS, 4)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)


def test_current_strategy_needs_the_regrets(tmp_path):
    """`tab:<file>@current` on a play-only blueprint (no regrets saved) silently played uniform random."""
    from headsup.blueprint import TabularBlueprint, TabularPlayer

    bp = TabularBlueprint(FHP, buckets=10, samples=50).fit_abstraction(2000, 0, 4)
    bp.run(2000, 1, 4)
    bp.save(tmp_path / "play.pt", play_only=True)
    bp.save(tmp_path / "full.pt")
    with pytest.raises(ValueError, match="regret"):
        TabularPlayer(str(tmp_path / "play.pt"), current=True)
    TabularPlayer(str(tmp_path / "play.pt"))  # the average strategy is what a play-only file has
    TabularPlayer(str(tmp_path / "full.pt"), current=True)


def test_public_states_are_grouped_exactly():
    """Rows were grouped by ONE rounded random projection of the public features: these two turn boards collided, so
    one of them was answered with the other's node and board."""
    from headsup.blueprint import public_groups
    from headsup.cards import CARD_FEATURES
    from headsup.engine import HeadsUpPoker

    e = HeadsUpPoker(game=DEFAULT_GAME)
    e.reset(list(range(40, 49)))
    for _ in range(4):
        e.step(1)  # check / call to the turn
    rows = np.repeat(e.observation()[None], 3, 0)
    for i, board in enumerate(([1, 26, 28, 13], [12, 14, 26, 4], [1, 26, 28, 13])):
        rows[i, 6:18] = CARD_FEATURES[board].reshape(12)
    first, inverse = public_groups(rows)
    assert len(first) == 2 and inverse[0] == inverse[2] != inverse[1]


def test_average_strategy_mode_is_selectable_and_recorded(tmp_path):
    """--strategy-every only matters for Pluribus's sampled action counters, which were not selectable from the
    command line (the default accumulates sigma at every sampled opponent infoset, where the option is inert)."""
    from headsup.blueprint import TabularBlueprint, main

    common = ["--game", "fhp", "--iterations", "2000", "--threads", "2", "--buckets", "8", "--samples", "20", "--situations", "400",
              "--chunks", "1", "--eval-hands", "0"]
    main(common + ["--out", str(tmp_path / "dense.pt")])
    main(common + ["--average", "counters", "--strategy-every", "7", "--out", str(tmp_path / "counters.pt")])
    dense, counters = TabularBlueprint.load(tmp_path / "dense.pt"), TabularBlueprint.load(tmp_path / "counters.pt")
    assert dense.params["dense_average"] is True
    assert counters.params["dense_average"] is False and counters.params["strategy_interval"] == 7
    assert np.asarray(counters.cpp.phi).sum() > 0 and np.asarray(dense.cpp.phi).sum() > 0


def test_bucket_of_a_situation_is_a_function_of_the_cards(tmp_path):
    """Pluribus: "each information situation is put into one of 200 buckets".  Ours drew a fresh Monte-Carlo hand
    strength at every lookup (the same hand on the same flop landed in its modal bucket 16 % of the time) and the
    1326-hand queries used another estimator than single ones."""
    from headsup.blueprint import TabularBlueprint

    bp = TabularBlueprint(DEFAULT_GAME, buckets=200, samples=500).fit_abstraction(20000, 0, 4)
    rng = np.random.default_rng(0)
    node = bp.node_of([1, 1])  # the first flop decision
    for nb, rnd in ((3, 1), (4, 2)):
        for _ in range(5):
            cards = [int(c) for c in rng.permutation(52)[: 2 + nb]]
            c0, c1, board = cards[0], cards[1], cards[2:]
            seen = {bp.cpp.bucket(rnd, c0, c1, board, seed) for seed in range(12)}
            assert len(seen) == 1
            assert bp.cpp.bucket(rnd, c1, c0, board[::-1], 99) in seen  # nor on the order of the cards
    # a query for all hands of a public state gives every hand the row of its own bucket
    board = [5, 20, 33]
    hands = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)], dtype=np.int32)
    all_rows = bp.strategy_for_hands(node, hands, board, seed=1)
    for i in rng.choice(len(hands), 25, replace=False):
        a, b = map(int, hands[i])
        if a in board or b in board:
            continue
        key = bp.cpp.bucket(1, a, b, board, 7)
        np.testing.assert_array_equal(all_rows[i], np.asarray(bp.cpp.strategy(node, key)))


def test_table_abstraction_is_the_same_in_every_process(tmp_path):
    """The exact per-board abstraction: the flop features come from all 1,081 completions by default (sampled ones
    were re-drawn by every process, so play and training could disagree about a flop's buckets)."""
    from headsup.blueprint import TabularBlueprint

    a = TabularBlueprint(DEFAULT_GAME, buckets=30, samples=100, mode="table").fit_abstraction(10 * 1326, 0, 4)
    assert a.completions == 0  # 0 = every completion
    a.save(tmp_path / "t.pt")
    b = TabularBlueprint.load(tmp_path / "t.pt")
    board = [5, 20, 33]
    for c0, c1 in ((0, 1), (12, 25), (40, 41)):
        assert a.cpp.bucket(1, c0, c1, board, 1) == b.cpp.bucket(1, c0, c1, board, 2)
    f = np.array(a.cpp.features(1, board, 1))
    np.testing.assert_array_equal(f, np.array(b.cpp.features(1, board, 5)))
    sampled = TabularBlueprint(DEFAULT_GAME, buckets=30, samples=100, mode="table", completions=50).fit_abstraction(10 * 1326, 0, 4)
    np.testing.assert_array_equal(np.array(sampled.cpp.features(1, board, 1)), np.array(sampled.cpp.features(1, board, 2)))  # seeded by the board
    assert np.abs(np.array(sampled.cpp.features(1, board, 1)) - f).max() > 1e-3  # and an approximation of the exact features


def test_pruning_threshold_scales_with_the_run_and_is_reported(tmp_path, capsys):
    """Pluribus's -300M is for its stacks and its billions of iterations; scaled by the stack alone it was never
    reached at tens of millions of iterations (most negative regret after 20 M: -1.4e6 against -3e6), so the
    "Linear MCCFR with pruning" blueprint never pruned.  The threshold now scales with the stack AND the
    iterations, and every chunk reports how many regrets are below it."""
    import json

    from headsup.blueprint import TabularBlueprint, main, prune_threshold

    assert prune_threshold(DEFAULT_GAME, 20_000_000) == pytest.approx(-3.0e5)  # ~ the 0.1 % quantile measured there
    assert prune_threshold(DEFAULT_GAME, 20_000_000, scale=5e-5) == pytest.approx(-1.0e5)
    assert prune_threshold(FHP, 1_000_000) == pytest.approx(-1.5e-4 * 100_000 * 1_000_000)
    common = ["--game", "fhp", "--iterations", "40000", "--threads", "2", "--buckets", "8", "--samples", "20", "--situations", "400",
              "--chunks", "2", "--eval-hands", "0"]
    main(common + ["--prune-scale", "1e-6", "--out", str(tmp_path / "p.pt")])
    out = capsys.readouterr().out
    bp = TabularBlueprint.load(tmp_path / "p.pt")
    assert bp.params["prune_threshold"] == pytest.approx(-1e-6 * 100_000 * 40_000)
    assert bp.params["regret_floor"] == pytest.approx(bp.params["prune_threshold"] * 310 / 300)
    log = json.load(open(str(tmp_path / "p.pt") + ".log.json"))
    assert 0.0 < log[-1]["prunable"] < 1.0 and "prunable" in out
    assert float(np.asarray(bp.cpp.regret).min()) >= bp.params["regret_floor"] - 1e-3  # the floor holds
    main(common + ["--no-prune", "--out", str(tmp_path / "n.pt")])
    assert json.load(open(str(tmp_path / "n.pt") + ".log.json"))[-1]["prunable"] == 0.0
