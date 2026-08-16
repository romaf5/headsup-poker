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
