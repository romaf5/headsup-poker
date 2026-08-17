"""Multi-street vector solver (public chance sampling, bucketed later rounds): exact best responses
against the solved subgame strategies via the hold'em best-response evaluator."""

import numpy as np
import pytest

from headsup import native
from headsup.algos.holdem_br import HoldemBestResponse
from headsup.engine import HeadsUpPoker
from headsup.game import DEFAULT_GAME
from headsup.lbr import NUM_COMBOS, valid_combos
from headsup.public import board_cards, hero_cards, replay_from_obs

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")

DECK = list(range(9))  # seat 0: 0,1  seat 1: 2,3  board 4..8
BOARD = [4, 5, 6, 7, 8]


class TreePlayer:
    """Answers strategy queries (hand-substituted observation rows of one public state) from a
    solver's tree: replays the observed actions to the node, maps hands to buckets via the board."""

    def __init__(self, solver, game, root_actions, current=False):
        self.sv, self.game, self.root_actions, self.current = solver, game, list(root_actions), current
        self.known = 0

    def probs(self, rows, ids=None):
        rows = np.asarray(rows)
        board = board_cards(rows[0])
        row = next(r for r in rows if not set(hero_cards(r)) & set(board))
        actions = []
        replay_from_obs(row, self.game, on_action=lambda e, seat, a: actions.append(int(a)))
        assert actions[: len(self.root_actions)] == self.root_actions
        node = 0
        for a in actions[len(self.root_actions):]:
            node = self.sv.child(node, a)
            assert node > 0
        drawn = board[self.known:]
        return self.sv.strategy_on_board(node, drawn, self.current)


def _roots(actions, deck=DECK):
    cpp = native.module()
    ce = cpp.Engine()
    ce.reset(deck)
    pe = HeadsUpPoker(rng=np.random.default_rng(0), game=DEFAULT_GAME)
    pe.reset(deck)
    for a in actions:
        ce.step(a)
        pe.step(a)
    return ce, pe


class UniformPlayer:
    def __init__(self, game):
        self.game = game

    def probs(self, rows, ids=None):
        from headsup.engine import legal_mask_from_obs

        legal = legal_mask_from_obs(np.asarray(rows, dtype=np.float32), self.game)
        return legal / legal.sum(1, keepdims=True)


def test_turn_subgame_solve_is_near_exact_with_all_rivers():
    """Turn root: the 48 possible rivers enumerated as the evaluator's boards -> exact best responses."""
    cpp = native.module()
    actions = [1, 1, 1, 1]  # limp / check, flop check / check: turn, BB to act
    ce, pe = _roots(actions)
    r = valid_combos(BOARD[:4]).astype(np.float32)
    sv = cpp.VectorSolver()
    sv.build(ce, 500)
    assert sv.root_round == 2 and sv.num_nodes > 100
    sv.set_ranges(r, r.copy())
    boards = [tuple(BOARD[:4] + [c]) for c in range(52) if c not in BOARD[:4]]
    br = HoldemBestResponse(UniformPlayer(DEFAULT_GAME), DEFAULT_GAME, boards=len(boards))
    ex_uniform = br.evaluate_from(pe, [r, r], boards)["exploitability_chips"]
    sv.run(300, 1, 1)
    tp = TreePlayer(sv, DEFAULT_GAME, actions)
    tp.known = 4
    br.player, br.probs_cache = tp, {}
    res = br.evaluate_from(pe, [r, r], boards)
    assert res["values"][0] == pytest.approx(-res["values"][1], abs=1e-9)
    assert res["br_values"][0] >= res["values"][0] - 1e-9 and res["br_values"][1] >= res["values"][1] - 1e-9
    assert res["exploitability_chips"] < 0.2 * ex_uniform, (res["exploitability_chips"], ex_uniform)
    # more iterations, multi-threaded: still improving (or at least as good)
    sv.run(1500, 2, 8)
    br.probs_cache = {}
    res2 = br.evaluate_from(pe, [r, r], boards)
    assert res2["exploitability_chips"] < 0.6 * res["exploitability_chips"], (res2["exploitability_chips"], res["exploitability_chips"])
    # the final-iterate strategy is a valid distribution too
    cur = sv.root_strategy(True)
    np.testing.assert_allclose(cur[r > 0].sum(1), 1.0, atol=1e-5)


def test_flop_subgame_solve_reduces_exploitability():
    cpp = native.module()
    actions = [1, 1]  # limp / check: flop, BB to act
    ce, pe = _roots(actions)
    r = valid_combos(BOARD[:3]).astype(np.float32)
    sv = cpp.VectorSolver()
    sv.build(ce, 200)
    assert sv.root_round == 1
    sv.set_ranges(r, r.copy())
    rng = np.random.default_rng(0)
    rest = [c for c in range(52) if c not in BOARD[:3]]
    boards = [tuple(BOARD[:3] + list(rng.choice(rest, 2, replace=False))) for _ in range(12)]
    br = HoldemBestResponse(UniformPlayer(DEFAULT_GAME), DEFAULT_GAME, boards=len(boards))
    ex_uniform = br.evaluate_from(pe, [r, r], boards)["exploitability_chips"]
    sv.run(300, 3, 8)
    tp = TreePlayer(sv, DEFAULT_GAME, actions)
    tp.known = 3
    br.player, br.probs_cache = tp, {}
    res = br.evaluate_from(pe, [r, r], boards)
    assert res["values"][0] == pytest.approx(-res["values"][1], abs=1e-9)
    assert res["exploitability_chips"] < 0.25 * ex_uniform, (res["exploitability_chips"], ex_uniform)  # ~1.4 vs 15.7 chips


def test_buckets_follow_equity_and_freeze_pins_a_hand():
    cpp = native.module()
    deck = [0, 1, 2, 3, 4, 18, 33, 47, 8]  # board 6s 7h 9d Tc Ts (a paired, rainbow-ish board)
    ce, _ = _roots([1, 1, 1, 1], deck)
    sv = cpp.VectorSolver()
    sv.build(ce, 100)
    board = deck[4:]
    b = np.array(sv.buckets_of(3, [board[4]]))  # river-round buckets on the complete board
    ok = valid_combos(board) > 0
    equity = np.array(cpp.BoardTable(board, 5).equity)
    assert b[ok].min() < 20 and b[ok].max() == 99  # quad tens (Th Td) are the nuts
    assert b[cpp.combo_index(21, 34)] == 99
    order = np.argsort(equity[ok])
    assert np.all(np.diff(b[ok][order]) >= 0)  # buckets are monotone in equity
    r = valid_combos(board[:4]).astype(np.float32)
    sv.set_ranges(r, r.copy())
    hh = cpp.combo_index(2, 3)
    sv.freeze(0, hh, 2)
    sv.run(50, 0, 1)
    assert list(sv.root_strategy()[hh]) == [0.0, 0.0, 1.0, 0.0]
    assert list(sv.root_strategy(True)[hh]) == [0.0, 0.0, 1.0, 0.0]
    with pytest.raises(RuntimeError):
        sv.freeze(0, hh, 0)  # BB has nothing to call: fold is illegal
    with pytest.raises(RuntimeError):
        cpp.VectorSolver().build(cpp.Engine())  # preflop root: three unknown cards


def test_board_table_matches_the_numpy_showdown_values():
    from headsup.cards import hand_strength
    from headsup.lbr import COMBOS
    from headsup.search import _opponent_mass, _showdown_values

    cpp = native.module()
    rng = np.random.default_rng(0)
    ok = valid_combos(BOARD)
    reach = np.where(ok, rng.random(NUM_COMBOS), 0.0)
    strength = np.full(NUM_COMBOS, np.inf)
    for h in np.flatnonzero(ok):
        strength[h] = hand_strength([int(COMBOS[h, 0]), int(COMBOS[h, 1])], BOARD)
    t = cpp.BoardTable(BOARD, 5)
    valid = ok > 0
    np.testing.assert_allclose(t.showdown_values(reach)[valid], _showdown_values(reach, strength)[valid], atol=1e-9)
    np.testing.assert_allclose(cpp.BoardTable.opponent_mass(reach), _opponent_mass(reach), atol=1e-9)
    assert np.array_equal(np.array(t.strength)[valid], strength[valid])
