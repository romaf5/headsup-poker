import numpy as np
import pytest
import torch

from headsup import native
from headsup.cards import card_from_str
from headsup.engine import HeadsUpPoker
from headsup.enums import Action
from headsup.game import DEFAULT_GAME
from headsup.lbr import COMBOS, NUM_COMBOS, LocalBestResponse, _Table, substitute_hands, valid_combos
from headsup.players import mask_illegal

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")


def c(*names):
    return [card_from_str(n) for n in names]


def test_combo_tables_and_substitution():
    cpp = native.module()
    assert len(COMBOS) == NUM_COMBOS == cpp.NUM_COMBOS
    for i in (0, 1, 500, 1325):  # index convention shared with the C++ equity kernel
        assert cpp.combo_index(int(COMBOS[i, 0]), int(COMBOS[i, 1])) == i
    ok = valid_combos(c("As", "Kd"))
    assert ok.sum() == 1326 - 51 - 50
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset(c("As", "Ad", "7c", "2d", "Kh", "Qs", "Jd", "3c", "9h"))
    rows = substitute_hands(e.observation(1))
    assert rows.shape == (NUM_COMBOS, 80)
    np.testing.assert_array_equal(rows[:, 6:], np.repeat(e.observation(1)[None, 6:], NUM_COMBOS, axis=0))
    assert rows[cpp.combo_index(*sorted(c("7c", "2d"))), :6].tolist() == e.observation(1)[:6].tolist()


def test_equity_kernel_matches_brute_force_on_river():
    from headsup.cards import hand_strength

    cpp = native.module()
    hero, board = c("As", "Ad"), c("Kh", "Qs", "Jd", "3c", "9h")
    eq = cpp.equity_vs_all(hero[0], hero[1], board, 200, 100, 0)
    mine = hand_strength(hero, board)
    for idx in np.random.default_rng(0).choice(NUM_COMBOS, 200, replace=False):
        a, b = COMBOS[idx]
        if a in hero + board or b in hero + board:
            assert eq[idx] == -1
            continue
        his = hand_strength([a, b], board)
        assert eq[idx] == (1.0 if mine < his else 0.5 if mine == his else 0.0)


def test_lbr_calls_a_shove_with_aces_and_folds_junk():
    lbr = LocalBestResponse("allin", num_tables=2, device="cpu", seed=0, duplicate=False, workers=2)
    for hand, expected in ((("As", "Ad"), Action.CHECK_CALL), (("7c", "2d"), Action.FOLD)):
        t = _Table(0, DEFAULT_GAME)
        t.start(1, c("Kh", "Kd") + c(*hand) + c("Qs", "Jd", "3c", "9h", "4h"))  # seat 0 = opponent, seat 1 = LBR
        assert t.engine.current == 0
        t.engine.step(Action.ALL_IN)  # opponent (SB) shoves; LBR is the big blind facing 98 to call
        lbr._lbr_step([t])
        taken = int(np.argmax(lbr.action_counts.sum(axis=0)))
        assert taken == expected, (hand, lbr.action_counts)
        lbr.action_counts[:] = 0


class _PairsOnlyPlayer:
    """Calls with pocket pairs, folds everything else (when facing a bet)."""

    def probs(self, obs, ids=None):
        obs = np.asarray(obs)
        pair = obs[:, 0] == obs[:, 3]  # same rank+1 in both hand slots
        p = np.zeros((len(obs), 4), dtype=np.float32)
        p[pair, Action.CHECK_CALL] = 1.0
        p[~pair, Action.FOLD] = 1.0
        return mask_illegal(p, obs)

    def __call__(self, obs, ids=None):
        return self.probs(obs).argmax(axis=1)


def test_range_update_uses_the_opponents_strategy():
    lbr = LocalBestResponse("call", num_tables=2, device="cpu", seed=0, duplicate=False, workers=2,
                            opponent=_PairsOnlyPlayer(), model=_PairsOnlyPlayer())
    t = _Table(0, DEFAULT_GAME)
    t.start(0, c("As", "Kd") + c("7c", "7d") + c("Qs", "Jd", "3c", "9h", "4h"))  # LBR = SB, opponent BB holds 77
    t.engine.step(Action.RAISE)  # LBR raises; the opponent now faces a bet
    before = t.range.copy()
    lbr._opponent_step([t])  # the opponent calls (it has a pair)
    pairs = COMBOS[:, 0] % 13 == COMBOS[:, 1] % 13
    assert np.all(t.range[~pairs] == 0) and t.range[pairs].sum() == pytest.approx(1.0)
    assert np.all(t.range[pairs & (before > 0)] > 0)
    assert t.range[valid_combos(c("As", "Kd")) == 0].sum() == 0  # blockers stay excluded


def test_thinned_bank_keeps_total_weight_and_is_identical_when_not_thinning():
    from headsup.model import BaseModel
    from headsup.sdcfr import IterateBank

    torch.manual_seed(0)
    nets = [[BaseModel() for _ in range(9)] for _ in range(2)]
    for seat in nets:
        for m in seat:
            torch.nn.init.normal_(m.action_head.weight, std=0.5)
    bank = IterateBank.from_state_dicts([[m.state_dict() for m in seat] for seat in nets], "cpu", nets[0][0].config)
    assert bank.thin(20) is bank
    thin = bank.thin(3)
    assert thin.T == 3 and float(thin.weights.sum()) == pytest.approx(float(bank.weights.sum()))
    obs = np.stack([HeadsUpPoker(rng=np.random.default_rng(1)).reset() for _ in range(4)])
    # the last iterate represents the last bin, so its strategy is one of the thinned ones
    np.testing.assert_allclose(thin.strategies(0, obs)[-1].numpy(), bank.strategies(0, obs)[-1].numpy())


def test_lbr_beats_simple_bots_quickly():
    lbr = LocalBestResponse("call", num_tables=16, device="cpu", seed=0, workers=4, mc_samples=50)
    r = lbr.play(60, progress=False)
    assert len(r) == 60 and r.mean() > 5  # ~+16 chips/hand vs the calling station
    lbr = LocalBestResponse("allin", num_tables=32, device="cpu", seed=0, workers=4, mc_samples=50)
    r = lbr.play(600, progress=False)  # duplicate pairs: the maniac's all-in luck largely cancels
    assert r.mean() > 2, r.mean()  # ~+8 chips/hand: folds junk, calls with strong hands
    assert lbr.action_counts[0, Action.FOLD] > 0 and lbr.action_counts[0, Action.CHECK_CALL] > 0


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_equity_uses_the_games_showdown_board():
    """FHP's showdown is on the flop: equity_vs_all(final_cards=3) must not deal a turn and river."""
    from headsup.cards import hand_strength

    cpp = native.module()
    board = [0, 18, 33]  # 2s 7h 9d
    hero = (12, 25)  # As Ah
    eq = np.asarray(cpp.equity_vs_all(hero[0], hero[1], board, 200, 100, 0, 3))
    s_me = hand_strength(list(hero), board)
    for a, b in [(1, 14), (5, 31), (18, 44), (26, 39)]:
        if a in board or b in board or a in hero or b in hero:
            continue
        s = hand_strength([a, b], board)
        exact = 1.0 if s_me < s else 0.5 if s_me == s else 0.0
        assert eq[cpp.combo_index(a, b)] == pytest.approx(exact)


def test_lbr_plays_the_same_number_of_hands_at_every_table():
    """play() returned the first pairs to finish among lock-step tables: short hands were over-represented and the
    hands still in flight were dropped (the bias env.play_hands had; +2 chips at 2 tables / 3000 hands vs 'call')."""
    lbr = LocalBestResponse("call", num_tables=8, device="cpu", seed=0, workers=4, mc_samples=20)
    starts = np.zeros(lbr.n, dtype=int)
    start = lbr._start_table

    def counting(t):
        starts[t.id] += 1
        start(t)

    lbr._start_table = counting
    r = lbr.play(30, progress=False)  # 4 duplicate pairs: 8 pairs of hands each, the first 30 results returned
    assert len(r) == 30 and starts.tolist() == [8] * 8


def test_transition_likelihood_counts_an_action_as_what_the_engine_executes():
    """A fold with nothing to call is executed as a check: a model that 'folds' half the time checks with
    probability 1 (illegal indices were skipped: 0.5, an under-counted likelihood for unmasked models)."""
    from headsup.lbr import transition_likelihood

    e = HeadsUpPoker(game=DEFAULT_GAME)
    e.reset(list(range(9)))
    e.step(Action.CHECK_CALL)  # the small blind calls: the big blind has nothing to call
    assert not e.legal_mask()[Action.FOLD]
    sigma = np.zeros((NUM_COMBOS, e.num_actions))
    sigma[:, Action.FOLD] = sigma[:, Action.CHECK_CALL] = 0.5
    np.testing.assert_allclose(transition_likelihood(e, Action.CHECK_CALL, sigma), 1.0)


def test_lbr_models_the_pluribus_bot_by_its_blueprint():
    """`pluribus[@options]` is a search player like `search:`: LBR queries its blueprint (the SearchPlayer itself was
    queried with 1326 hand-substituted rows as if they were tables: KeyError on the first bet lookahead)."""
    from headsup.blueprint import TabularPlayer
    from headsup.lbr import _model_for
    from headsup.paths import DEFAULT_BLUEPRINT_PATH

    import os

    if not os.path.exists(DEFAULT_BLUEPRINT_PATH):
        pytest.skip("the shipped blueprint is not present")
    for spec in ("pluribus", "pluribus@it20@th2@b50"):
        assert isinstance(_model_for(spec, "cpu", 0), TabularPlayer)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_flop_equities_are_enumerated_by_default():
    """Lisy & Bowling compute the rollout values exhaustively after the flop; the default sampled 200 of the flop's
    1081 runouts (turn: all 46).  Enumerated equities do not depend on the seed."""
    a = LocalBestResponse("call", num_tables=2, device="cpu", seed=0, duplicate=False, workers=2)
    b = LocalBestResponse("call", num_tables=2, device="cpu", seed=1, duplicate=False, workers=2)
    sampled = LocalBestResponse("call", num_tables=2, device="cpu", seed=1, duplicate=False, workers=2, max_exact=100)
    deck = [51, 38, 0, 13, 5, 18, 31, 44, 9]
    for lbr in (a, b, sampled):
        t = lbr.tables[0]
        t.start(1, deck)
        t.engine.step(Action.CHECK_CALL)
        t.engine.step(Action.CHECK_CALL)
        assert t.engine.stage == 1 and t.engine.current == 1
    ea, eb, es = (lbr._equities([lbr.tables[0]])[0] for lbr in (a, b, sampled))
    assert np.array_equal(ea, eb)
    assert not np.array_equal(ea, es) and np.abs(ea - es)[ea >= 0].max() < 0.2


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_lbr_can_check_call_until_a_given_round():
    """The LBR paper's Table 2: LBR that only check/calls in the first rounds (its myopic pre-flop raises and folds
    often cost more than they win) - there the strongest settings against most bots."""
    kw = dict(num_tables=16, device="cpu", seed=0, workers=4, mc_samples=50, max_exact=100, duplicate=False)
    lbr = LocalBestResponse("allin", from_round=1, **kw)
    r = lbr.play(64, progress=False)
    counts = lbr.action_counts
    assert counts[0].sum() > 0 and counts[0, Action.CHECK_CALL] == counts[0].sum()  # it calls every shove
    assert lbr.summary(r)["from_round"] == "flop"
    default = LocalBestResponse("allin", **kw)
    default.play(64, progress=False)
    assert default.action_counts[0, Action.FOLD] > 0 and default.summary(r)["from_round"] == "preflop"
    with pytest.raises(ValueError, match="round"):
        LocalBestResponse("allin", from_round=4, **kw)
