import numpy as np
import pytest

from headsup.cards import card_from_str, hand_strength
from headsup.engine import OBS_DIM, HeadsUpPoker
from headsup.enums import Action, Stage


def deck_from(*cards):
    """Build a 9-card deal: seat0 hand, seat1 hand, board."""
    ids = [card_from_str(c) for c in cards]
    assert len(ids) == 9 and len(set(ids)) == 9
    return ids


AA_vs_72 = deck_from("As", "Ad", "7c", "2d", "Kh", "Qs", "Jd", "3c", "9h")


def test_reset_blinds_and_first_to_act():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    obs = e.reset(AA_vs_72)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert e.current == 0  # dealer / small blind acts first pre-flop
    assert e.stacks == [99, 98] and e.bets == [1, 2] and e.pot == 3
    assert e.stage == Stage.PREFLOP and not e.done


def test_observation_layout():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    obs = e.reset(AA_vs_72)
    # As: rank 12 suit 0 -> (13, 1, 13); Ad: rank 12 suit 2 -> (13, 3, 39)
    assert obs[0:6].tolist() == [13, 1, 13, 13, 3, 39]
    assert obs[6:21].tolist() == [0] * 15  # no board pre-flop
    assert obs[21] == 0 and obs[22] == 0  # stage, first_to_act_next_stage (seat 0 = dealer)
    np.testing.assert_allclose(obs[23:31], [1 / 3, 1 / 3, 2 / 3, 1 / 3, 2 / 3, 33, 0.003, 1 / 99], rtol=1e-6)


def test_fold_rewards_are_zero_sum():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(AA_vs_72)
    _, rewards, done, _ = e.step(Action.FOLD)
    assert done and rewards == [-1, 1] and e.folded == 0


def test_limp_check_goes_to_flop_and_bb_acts_first():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(AA_vs_72)
    e.step(Action.CHECK_CALL)  # SB limps
    assert e.current == 1 and e.stage == Stage.PREFLOP  # BB has the option
    e.step(Action.CHECK_CALL)  # BB checks
    assert e.stage == Stage.FLOP and e.current == 1 and e.stage_bets == [0, 0]
    assert len(e.visible_board) == 3


def test_showdown_all_in_call():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(AA_vs_72)
    e.step(Action.ALL_IN)
    assert e.stacks[0] == 0 and e.bets[0] == 100
    _, rewards, done, _ = e.step(Action.CHECK_CALL)
    assert done and e.stage == Stage.END
    assert rewards == [100, -100]  # aces hold on K Q J 3 9
    assert hand_strength(e.hands[0], e.board) < hand_strength(e.hands[1], e.board)


def test_min_raise_and_raise_cap_becomes_all_in():
    e = HeadsUpPoker(rng=np.random.default_rng(0), raise_cap=3)
    e.reset(AA_vs_72)
    e.step(Action.RAISE)  # SB raises to 4
    assert e.stage_bets == [4, 2] and e.stacks == [96, 98]
    e.step(Action.RAISE)  # BB re-raises to 6
    assert e.stage_bets == [4, 6]
    e.step(Action.RAISE)  # third consecutive raise -> all-in
    assert e.stacks[0] == 0 and e.stage_bets[0] == 100
    e.step(Action.FOLD)
    assert e.done and e.rewards == [6, -6]


def test_raise_short_stack_becomes_all_in_and_showdown_pays_min_bet():
    e = HeadsUpPoker(rng=np.random.default_rng(0), stack_size=5)
    e.reset(AA_vs_72)
    e.step(Action.RAISE)  # SB: call 1 + raise 2 = 3 -> 4 total, stack 1
    e.step(Action.RAISE)  # BB: needs 2 + 2 = 4 but has 3 -> all-in for 5 total
    assert e.stacks == [1, 0] and e.bets == [4, 5]
    _, rewards, done, _ = e.step(Action.CHECK_CALL)  # SB calls the last chip
    assert done and rewards == [5, -5]


def test_river_check_check_ends_hand():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(AA_vs_72)
    for _ in range(4):  # preflop limp/check, flop, turn, river check/check
        e.step(Action.CHECK_CALL)
        _, rewards, done, _ = e.step(Action.CHECK_CALL)
    assert done and rewards == [2, -2]


def test_fold_with_nothing_to_call_is_a_check():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(AA_vs_72)
    e.step(Action.CHECK_CALL)  # SB limps
    assert not e.fold_allowed and Action.FOLD not in e.legal_actions()
    _, _, done, _ = e.step(Action.FOLD)  # BB "folds" facing no bet -> treated as check
    assert not done and e.stage == Stage.FLOP and e.folded == -1
    assert e.fold_allowed is False  # BB first to act on the flop, nothing to call
    e.step(Action.RAISE)  # BB bets 2
    assert e.fold_allowed  # SB now faces a bet


def test_clone_is_independent():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(AA_vs_72)
    c = e.clone()
    c.step(Action.ALL_IN)
    assert e.stacks == [99, 98] and c.stacks[0] == 0
    assert c.hands is e.hands  # cards are shared, chips are not


def test_random_play_invariants():
    rng = np.random.default_rng(1)
    e = HeadsUpPoker(rng=rng)
    for _ in range(2000):
        e.reset()
        while not e.done:
            assert e.pot == sum(e.bets) and all(s >= 0 for s in e.stacks)
            assert e.stacks[0] + e.bets[0] == 100 and e.stacks[1] + e.bets[1] == 100
            e.step(rng.integers(4))
        assert sum(e.rewards) == 0
        assert abs(e.rewards[0]) <= min(e.bets) or e.folded >= 0


def test_invalid_action():
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset()
    with pytest.raises(ValueError):
        e.step(7)
