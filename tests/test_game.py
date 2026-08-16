"""Configurable action trees (bet sizes): game config, engines, masks, traversal, players."""

import numpy as np
import pytest
import torch

from headsup import native
from headsup.engine import HeadsUpPoker, legal_mask_from_obs
from headsup.enums import Action
from headsup.game import DEFAULT_GAME, GameConfig, action_label, parse_bet_sizes
from headsup.model import BaseModel

POT_GAME = GameConfig(bet_sizes=(0.5, 1.0, 2.0), mask_redundant=True)
MIXED_GAME = GameConfig(bet_sizes=("min", 1.0), raise_cap=4, mask_redundant=True)


def test_game_config_basics():
    assert parse_bet_sizes("min") == ("min",) and parse_bet_sizes("0.5, 1,2") == (0.5, 1.0, 2.0)
    assert parse_bet_sizes(["min", 1]) == ("min", 1.0)
    with pytest.raises(ValueError):
        parse_bet_sizes("0.25,0.5,0.75,1,2,3")
    g = POT_GAME
    assert g.num_actions == 6 and g.all_in == 5 and g.is_raise(2) and g.is_raise(4) and not g.is_raise(5)
    assert DEFAULT_GAME.num_actions == 4 and DEFAULT_GAME.tree_dict() == {"bet_sizes": ["min"], "raise_cap": 3, "mask_redundant": False}
    assert GameConfig.from_dict(g.tree_dict()) == g and g.with_(stack_size=200).stack_size == 200
    # SB opens: pot 3, 1 to call: half pot = 1 + 2, pot = 1 + 4, 2x pot = 1 + 8; never below a min-raise
    assert [g.raise_amount(a, 1, 3, 99) for a in (2, 3, 4)] == [3, 5, 9]
    assert GameConfig(bet_sizes=(0.1,)).raise_amount(2, 1, 3, 99) == 3  # 0.1 * 4 rounds to 0 -> min-raise 1 + 2
    assert [action_label(g, a) for a in range(6)] == ["fold", "call", "raise_0.5p", "raise_1p", "raise_2p", "allin"]
    assert action_label(DEFAULT_GAME, 2) == "raise" and action_label(MIXED_GAME, 2) == "raise_min"


def test_legal_mask_and_twins():
    g = POT_GAME
    mask, twins = g.legal_mask(1, 3, 99, 0, with_twins=True)
    assert mask == [True] * 6 and twins == list(range(6))
    mask, twins = g.legal_mask(0, 4, 98, 0, with_twins=True)  # nothing to call: no fold (twin = check)
    assert mask[0] is False and twins[0] == 1
    # short stack: pot and 2x pot reach the stack -> only the half-pot raise stays (others = all-in)
    mask, twins = g.legal_mask(1, 3, 5, 0, with_twins=True)
    assert mask == [True, True, True, False, False, True] and twins == [0, 1, 2, 5, 5, 5]
    # cap: the next raise would be the 3rd -> all raises masked with twin all-in
    mask, twins = g.legal_mask(24, 36, 70, 2, with_twins=True)
    assert mask[2:5] == [False] * 3 and twins[2:5] == [5, 5, 5]
    # duplicate sizes: 'min' and 1 pot coincide when 1 * (pot + call) rounds to the big blind
    mask, twins = MIXED_GAME.legal_mask(0, 2, 98, 0, with_twins=True)  # pot 2, bet 1x pot = 2 = min-raise 2
    assert mask == [False, True, True, False, True] and twins == [1, 1, 2, 2, 4]
    # the default game never masks raises (historic tree)
    assert DEFAULT_GAME.legal_mask(24, 36, 70, 2) == [True, True, True, True]


@pytest.mark.parametrize("game", [POT_GAME, MIXED_GAME, GameConfig(bet_sizes=(0.5, 1.0, 2.0), mask_redundant=False)])
def test_engine_multi_size_invariants_and_masks_from_obs(game):
    rng = np.random.default_rng(3)
    e = HeadsUpPoker(rng=rng, game=game)
    for _ in range(500):
        e.reset()
        while not e.done:
            legal = e.legal_mask()
            np.testing.assert_array_equal(legal_mask_from_obs(e.observation()[None], game)[0], legal)
            assert legal[1] and legal[game.all_in] and legal[0] == e.fold_allowed
            a = rng.choice(np.flatnonzero(legal))
            e.step(int(a))
            assert e.pot == sum(e.bets) and all(s >= 0 for s in e.stacks)
        assert sum(e.rewards) == 0


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
@pytest.mark.parametrize("game", [POT_GAME, MIXED_GAME])
def test_cpp_engine_matches_python_engine_multi_size(game):
    cpp = native.module()
    rng = np.random.default_rng(5)
    py = HeadsUpPoker(rng=rng, game=game)
    ce = cpp.Engine(native.engine_config(game=game))
    for _ in range(1500):
        perm = rng.permutation(52)
        o_py = py.reset(deck=perm)
        ce.reset(perm.tolist())
        while True:
            assert py.current == ce.current
            np.testing.assert_array_equal(o_py, ce.observation())
            assert list(ce.legal_mask()) == py.legal_mask()
            for a in range(2, game.all_in):
                assert ce.raise_amount(a) == py.raise_amount(a)
            a = int(rng.integers(game.num_actions))  # illegal actions are executed as their duplicates by both
            o_py, r_py, d_py, _ = py.step(a)
            d_c = ce.step(a)
            assert d_py == d_c
            if d_py:
                assert list(r_py) == list(ce.rewards)
                break


def _biased(bias, game, **config):
    m = BaseModel(game=game, **config)
    with torch.no_grad():
        m.action_head.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return m.numpy_weights()


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
@pytest.mark.parametrize("game", [POT_GAME, MIXED_GAME])
def test_cpp_traversal_matches_python_multi_size(game):
    from headsup.deepcfr.traverse import run_traversals_python

    cpp = native.module()
    rng = np.random.default_rng(0)
    n = 40
    decks = np.stack([rng.permutation(52)[:9] for _ in range(n)]).astype(np.int32)
    k = game.num_actions
    # deterministic strategies (C++ / Python RNG streams differ): seat 0 always check/calls (the
    # only positive advantage, always legal); seat 1 has no positive advantage and its argmax
    # fallback prefers the pot-size raise, then check/call when that raise is masked
    b0 = [0.0, 5.0] + [0.0] * (k - 2)
    b1 = [-3.0, -2.0, -4.0, -1.0, -5.0, -6.0][:k]
    w0, w1 = _biased(b0, game, features="history"), _biased(b1, game, features="history", rm_fallback="argmax")
    for traverser in (0, 1):
        pa, ps, pn = run_traversals_python([w0, w1], traverser, n, 2.0, 1, None, decks)
        out = cpp.run_traversals(native.make_model(w0), native.make_model(w1), traverser, n, 2.0, 1, native.engine_config(game=game), decks)
        assert pn == out[6] and len(pa) == len(out[1]) and len(ps) == len(out[4])
        assert pa.target.shape[1] == out[2].shape[1] == k
        np.testing.assert_array_equal(pa.obs, out[0])
        np.testing.assert_allclose(pa.target, out[2], atol=1e-4)
        np.testing.assert_allclose(ps.target, out[5], atol=1e-5)
        assert np.all(ps.target[ps.obs[:, 23] <= 0, 0] == 0)


@pytest.mark.parametrize("game", [POT_GAME])
def test_variant_models_with_more_actions_agree(game):
    from headsup.numpy_model import NumpyModel

    torch.manual_seed(1)
    m = BaseModel(game=game, features="both", arch="paper")
    with torch.no_grad():
        torch.nn.init.normal_(m.action_head.weight, std=0.5)
    assert m.num_actions == 6 and m.action_head.out_features == 6
    e = HeadsUpPoker(rng=np.random.default_rng(0), game=game)
    rows = []
    while len(rows) < 300:
        e.reset()
        while not e.done:
            rows.append(e.observation())
            e.step(int(np.random.default_rng(len(rows)).choice(np.flatnonzero(e.legal_mask()))))
    obs = np.stack(rows)
    with torch.no_grad():
        ref = m(torch.from_numpy(obs)).numpy()
    assert ref.shape == (300, 6)
    np.testing.assert_allclose(NumpyModel(m.numpy_weights())(obs), ref, atol=1e-4)
    if native.available():
        cm = native.make_model(m.numpy_weights())
        assert cm.num_actions == 6
        np.testing.assert_allclose(cm.forward(obs), ref, atol=1e-4)


def test_players_and_envs_in_a_multi_size_game():
    from headsup.env import PokerVecEnv, make_vec_env, play_hands
    from headsup.players import AlwaysAllInPlayer, RandomPlayer, RegretMatchingPlayer, TorchPolicyPlayer, make_player

    game = POT_GAME
    assert AlwaysAllInPlayer(game=game).action == 5 and make_player("allin", game=game).action == 5
    rp = RandomPlayer(seed=0, game=game)
    e = HeadsUpPoker(rng=np.random.default_rng(0), game=game)
    e.reset()
    p = rp.probs(e.observation()[None])[0]
    assert p.shape == (6,) and np.allclose(p, 1 / 6)  # SB facing the blind: everything legal
    e.step(Action.CHECK_CALL)  # limp: the BB has nothing to call -> no fold
    p = rp.probs(e.observation()[None])[0]
    assert p[0] == 0 and np.allclose(p[1:], 1 / 5)
    torch.manual_seed(0)
    model = BaseModel(game=game)
    with torch.no_grad():
        torch.nn.init.normal_(model.action_head.weight, std=0.5)
    agent = TorchPolicyPlayer(model, device="cpu", seed=0)
    assert agent.game == game
    env = make_vec_env(64, "call", seed=0, game=game)  # simple bot opponent: the env needs the agent's tree
    assert env.game.num_actions == 6
    r = play_hands(env, agent, 300)
    assert len(r) == 300
    env = PokerVecEnv(32, agent, seed=1)  # network opponent: the env takes its tree
    assert env.game == game
    r = play_hands(env, RandomPlayer(seed=2, game=game), 200)
    assert len(r) == 200
    if native.available():
        env = make_vec_env(64, model, seed=0)  # native env with a 6-action model opponent
        assert env.game.num_actions == 6
        r = play_hands(env, RegretMatchingPlayer([model, model], device="cpu", seed=0), 300)
        assert len(r) == 300
    with pytest.raises(ValueError):  # mismatching trees are refused
        play_hands(make_vec_env(8, "call", seed=0), agent, 10)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_lbr_and_trainer_smoke_multi_size(tmp_path):
    from headsup.deepcfr.train import main as train_main
    from headsup.lbr import LocalBestResponse

    lbr = LocalBestResponse("call", num_tables=8, device="cpu", seed=0, workers=2, mc_samples=30, game=POT_GAME)
    assert lbr.num_actions == 6
    r = lbr.play(20, progress=False)
    assert len(r) == 20 and lbr.action_counts.shape == (4, 6)
    train_main([
        "--algo", "both", "--iterations", "2", "--traversals", "30", "--workers", "2", "--device", "cpu", "--no-compile",
        "--bet-sizes", "0.5,1", "--features", "history", "--adv-capacity", "20000", "--strat-capacity", "20000",
        "--value-steps", "3", "--batch-size", "64", "--policy-epochs", "1", "--eval-hands", "0", "--eval-every", "0",
        "--policy-eval-every", "0", "--lbr-final-hands", "0", "--no-tensorboard", "--out", str(tmp_path),
    ])
    from headsup.players import make_player

    pol = make_player(f"cfr:{tmp_path / 'policy.pth'}", device="cpu", seed=0)
    assert pol.game.bet_sizes == (0.5, 1.0) and pol.game.mask_redundant and pol.game.num_actions == 5
    sd = make_player(f"sdcfr:{tmp_path / 'iterates.pt'}", device="cpu", seed=0)
    assert sd.game.num_actions == 5 and sd.bank.T == 3
