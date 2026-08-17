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
