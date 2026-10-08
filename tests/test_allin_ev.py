"""All-in EV: the expected result over the board cards still to come when the betting closed, instead of the one
runout that was dealt (same expectation, far less variance)."""

import itertools

import numpy as np
import pytest

from headsup import native
from headsup.cards import hand_strength

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")


def _brute_force(h0, h1, board, final_cards=5):
    """(P(seat 0 wins), P(tie)) by enumerating every completion of the board with the reference evaluator."""
    rest = [c for c in range(52) if c not in set(h0) | set(h1) | set(board)]
    win = tie = n = 0
    for extra in itertools.combinations(rest, final_cards - len(board)):
        full = list(board) + list(extra)
        s0, s1 = hand_strength(h0, full), hand_strength(h1, full)
        win += s0 < s1
        tie += s0 == s1
        n += 1
    return win / n, tie / n


def test_fast_seven_card_evaluator_matches_the_reference():
    """eval7 is one table lookup (flush by suit mask, else by rank multiset); the tables are built from the
    21-combination reference evaluator."""
    cpp = native.module()
    rng = np.random.default_rng(0)
    hands = [rng.permutation(52)[:7] for _ in range(20000)]
    for suited in (5, 6, 7):  # flushes and straight flushes are rare in random hands
        for _ in range(3000):
            suit = int(rng.integers(4))
            ranks = rng.permutation(13)[:suited]
            others = rng.permutation([c for c in range(52) if c // 13 != suit])[: 7 - suited]
            hands.append(np.concatenate([ranks + 13 * suit, others]))
    hands.append(np.array([8, 9, 10, 11, 12, 20, 33]))  # royal flush
    hands.append(np.array([12, 0, 1, 2, 3, 20, 33]))  # the wheel, suited
    for h in hands:
        cards = [int(c) for c in h]
        assert cpp.eval7(cards) == cpp.eval7_reference(cards), cards


def test_showdown_equity_is_the_exact_average_over_the_runouts():
    cpp = native.module()
    rng = np.random.default_rng(1)
    for n_board in (3, 4, 5):
        for _ in range(6 if n_board == 3 else 20):
            cards = [int(c) for c in rng.permutation(52)[: 4 + n_board]]
            h0, h1, board = cards[:2], cards[2:4], cards[4:]
            win, tie = cpp.showdown_equity(h0, h1, board, 5)
            want = _brute_force(h0, h1, board)
            assert (win, tie) == pytest.approx(want, abs=1e-12)
            back = cpp.showdown_equity(h1, h0, board, 5)
            assert back[1] == pytest.approx(tie, abs=1e-12) and back[0] == pytest.approx(1 - win - tie, abs=1e-12)
    # flop hold'em shows down on three cards: a pre-flop all-in there has 17,296 flops to come
    h0, h1 = [12, 25], [11, 24]  # aces against kings
    win, tie = cpp.showdown_equity(h0, h1, [], 3)
    assert (win, tie) == pytest.approx(_brute_force(h0, h1, [], 3), abs=1e-12)
    # pre-flop on five cards: 1,712,304 boards, exact; sampled runouts agree within their error
    win, tie = cpp.showdown_equity(h0, h1, [], 5)
    assert 0.80 < win < 0.84 and 0.0 < tie < 0.01  # aces are about 82 % against kings
    mc = cpp.showdown_equity(h0, h1, [], 5, 40000, 7)
    assert mc[0] == pytest.approx(win, abs=0.01)
    assert cpp.showdown_equity(h0, h1, [], 5, 40000, 7) == mc  # seeded
    with pytest.raises((RuntimeError, ValueError), match="cards"):
        cpp.showdown_equity([12, 25], [12, 24], [], 5)


def _play(engine, deck, actions):
    engine.reset(deck)
    for a in actions:
        engine.step(a)
    return engine


@pytest.mark.parametrize("line, stage", [([3, 1], 0), ([1, 1, 3, 1], 1), ([1, 1, 1, 1, 3, 1], 2)])
def test_engine_allin_ev_is_the_expectation_over_the_runouts(line, stage):
    """Both engines: an all-in called before the river is worth the stake times (P(win) - P(lose)) over the boards to come."""
    from headsup.engine import HeadsUpPoker
    from headsup.game import DEFAULT_GAME

    cpp = native.module()
    rng = np.random.default_rng(stage)
    for _ in range(3):
        deck = [int(c) for c in rng.permutation(52)[:9]]
        py = _play(HeadsUpPoker(game=DEFAULT_GAME), deck, line)
        cc = _play(cpp.Engine(native.engine_config(game=DEFAULT_GAME)), deck, line)
        assert py.done and cc.done and py.showdown_stage == cc.showdown_stage == stage
        known = deck[4:9][: (0, 3, 4)[stage]]
        if stage == 0:
            win, tie = cpp.showdown_equity(deck[0:2], deck[2:4], known, 5)
        else:
            win, tie = _brute_force(deck[0:2], deck[2:4], known)
        want = 100 * (win - (1 - win - tie))  # both stacks are in
        assert py.allin_ev() == pytest.approx([want, -want], abs=1e-9)
        assert cc.allin_ev() == pytest.approx([want, -want], abs=1e-9)
        assert abs(want) <= 100 and sorted(map(abs, py.rewards)) in ([0, 0], [100, 100])  # a hand can be drawing dead


def test_allin_ev_is_the_actual_result_when_nothing_was_left_to_chance():
    from headsup.engine import HeadsUpPoker
    from headsup.game import DEFAULT_GAME, FHP

    cpp = native.module()
    deck = list(range(9))
    for game, line in ((DEFAULT_GAME, [3, 0]), (DEFAULT_GAME, [1] * 8), (DEFAULT_GAME, [1, 1, 1, 1, 1, 1, 3, 1]), (FHP, [1, 1, 1, 1])):
        py = _play(HeadsUpPoker(game=game), deck, line)  # a fold; a checked-down river; a river all-in; a limit game
        cc = _play(cpp.Engine(native.engine_config(game=game)), deck, line)
        assert py.done and py.allin_ev() == [float(r) for r in py.rewards] == cc.allin_ev()
    e = HeadsUpPoker(game=DEFAULT_GAME)
    e.reset(deck)
    assert e.allin_ev() == [0.0, 0.0]  # a hand in progress has no result yet


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_play_hands_with_allin_ev_has_the_same_mean_and_less_variance(backend):
    """Calling station against the all-in bot: every hand is a pre-flop all-in. The EV results average to the same
    number as the dealt ones, with a fraction of the spread."""
    from headsup.env import make_vec_env, play_hands
    from headsup.players import AlwaysCallPlayer

    n = 6000 if backend == "cpp" else 600
    raw = play_hands(make_vec_env(64, "allin", seed=0, backend=backend), AlwaysCallPlayer(), n)
    ev = play_hands(make_vec_env(64, "allin", seed=0, backend=backend), AlwaysCallPlayer(), n, allin_ev=True)
    assert len(ev) == n and set(np.abs(raw).tolist()) <= {0.0, 100.0} and np.abs(ev).max() < 100
    assert ev.std() < 0.6 * raw.std()
    se = np.sqrt(raw.var() / n + ev.var() / n)
    assert abs(ev.mean() - raw.mean()) < 4 * se
    if backend == "cpp":  # the same deals in both modes: a sampled EV has the sign of the favourite far more often than not
        assert np.mean(np.sign(ev) == np.sign(raw)) > 0.55


def test_evaluation_tools_use_allin_ev_unless_told_not_to():
    from headsup.compare import head_to_head
    from headsup.deepcfr.evaluate import evaluate
    from headsup.lbr import LocalBestResponse
    from headsup.players import AlwaysCallPlayer

    m_ev, se_ev = head_to_head("call", "allin", 4000, num_envs=64, seed=0)
    m_raw, se_raw = head_to_head("call", "allin", 4000, num_envs=64, seed=0, allin_ev=False)
    assert se_ev < 0.6 * se_raw and abs(m_ev - m_raw) < 4 * np.hypot(se_ev, se_raw)
    ev = evaluate(AlwaysCallPlayer(), 2000, opponents=("allin",), num_envs=64, seed=0)["allin"]
    raw = evaluate(AlwaysCallPlayer(), 2000, opponents=("allin",), num_envs=64, seed=0, allin_ev=False)["allin"]
    assert ev != raw and round(raw * 2000) % 100 == 0  # dealt results are whole stacks
    whole = lambda r: np.abs(2 * r - np.round(2 * r)) < 1e-9  # a pair of dealt results is a multiple of half a chip
    lbr = LocalBestResponse("allin", num_tables=8, device="cpu", seed=0, workers=2, mc_samples=20)
    r = lbr.play(40, progress=False)
    assert lbr.allin_ev and not whole(r).all()  # expectations over the runouts
    lbr = LocalBestResponse("allin", num_tables=8, device="cpu", seed=0, workers=2, mc_samples=20, allin_ev=False)
    assert whole(lbr.play(40, progress=False)).all()


def test_lbr_values_showdowns_against_the_opponents_whole_range():
    """LBR knows the opponent's strategy, hence its range at a showdown: the result of a showdown is the expectation
    over that range (and over the cards to come), not the hand the opponent happened to hold. The same mean - the
    hand it holds is drawn from exactly that range - with less variance; only for an exact opponent model."""
    from headsup.lbr import LocalBestResponse

    kw = dict(num_tables=32, device="cpu", seed=0, workers=4, mc_samples=50, duplicate=False)
    plain = LocalBestResponse("random", range_ev=False, **kw)
    raw = plain.play(400, progress=False)
    ranged = LocalBestResponse("random", **kw)
    assert ranged.range_ev and not plain.range_ev
    ev = ranged.play(400, progress=False)
    assert ev.std() < 0.8 * raw.std()  # measured: 24 -> 13 chips per hand (65 dealt)
    assert abs(ev.mean() - raw.mean()) < 4 * np.sqrt(raw.var() / len(raw) + ev.var() / len(ev))
    assert ranged.summary(ev)["range_ev"] is True
    assert LocalBestResponse("random", allin_ev=False, **kw).range_ev is False  # --raw: dealt results
    assert LocalBestResponse("random", model_iterates=8, **kw).range_ev is False  # an approximate opponent model
