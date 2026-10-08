import numpy as np

from headsup.engine import OBS_DIM
from headsup.env import PokerVecEnv, SingleAgentEnv, play_hands
from headsup.players import AlwaysCallPlayer, RandomPlayer, make_player, sample_actions


def test_single_env_api_and_seat_alternation():
    env = SingleAgentEnv(AlwaysCallPlayer(), seed=0)
    seats = []
    for _ in range(4):
        obs = env.reset()
        seats.append(env.agent_seat)
        assert obs.shape == (OBS_DIM,)
        done = False
        while not done:
            obs, reward, done, info = env.step(1)
        assert env.engine.done
    assert seats == [0, 1, 0, 1]


def test_vec_env_auto_reset_and_reward_bookkeeping():
    env = PokerVecEnv(64, RandomPlayer(seed=1), seed=0)
    obs = env.reset()
    assert obs.shape == (64, OBS_DIM)
    total = 0
    for _ in range(50):
        obs, r, d, infos = env.step(np.ones(64, dtype=np.int64))
        assert obs.shape == (64, OBS_DIM) and r.shape == (64,) and d.shape == (64,)
        assert np.all(r[~d] == 0)
        total += int(d.sum())
    assert total == env.hands_completed > 0


def test_open_fold_during_reset_is_paid_on_next_step():
    class FoldBot:
        def __call__(self, obs):
            return np.zeros(len(obs), dtype=np.int64)

    env = SingleAgentEnv(FoldBot(), seat_mode="alternate", seed=0)
    env.reset()  # agent seat 0 (dealer): agent raises, bot (BB) folds
    _, reward, done, _ = env.step(2)
    assert done and reward == 2
    env.reset()  # agent seat 1: bot open-folds its small blind during reset
    assert env.engine.done
    _, reward, done, _ = env.step(0)  # any action collects the pot
    assert done and reward == 1


def test_play_hands_returns_requested_count():
    env = PokerVecEnv(32, AlwaysCallPlayer(), seed=0)
    r = play_hands(env, RandomPlayer(seed=0), 500)
    assert len(r) == 500


def test_sample_actions():
    rng = np.random.default_rng(0)
    probs = np.array([[0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    assert sample_actions(probs, rng, deterministic=True).tolist() == [1, 0]
    draws = np.array([sample_actions(probs, rng)[1] for _ in range(400)])
    assert set(draws.tolist()) == {0, 1}


def test_make_player_specs():
    obs = np.zeros((3, OBS_DIM), dtype=np.float32)
    assert make_player("call")(obs).tolist() == [1, 1, 1]
    assert make_player("allin")(obs).tolist() == [3, 3, 3]
    assert make_player("cfr", device="cpu", seed=0)(obs).shape == (3,)


class _FoldHalfOrCall:
    """As small blind folds its first decision with probability 1/2, otherwise check/calls: against a calling
    opponent that is exactly -0.25 chips/hand (it is the small blind in half of the hands and folds half of those)."""

    def __init__(self, seed, game):
        self.rng, self.game = np.random.default_rng(seed), game

    def __call__(self, obs, ids=None):
        first_sb = (obs[:, 21] == 0) & (obs[:, 22] == 0) & (obs[:, 32] == 0)  # pre-flop, small blind, no action yet
        return np.where(first_sb & (self.rng.random(len(obs)) < 0.5), 0, 1).astype(np.int64)


def test_play_hands_does_not_favour_short_hands():
    """Every table contributes the same number of hands: taking the first N hands to finish across 1024 parallel
    tables over-represented the short ones (here the folds: -0.32 instead of -0.25 with 2 hands per table)."""
    from headsup.env import make_vec_env
    from headsup.game import DEFAULT_GAME

    means = []
    for seed in range(20):
        r = play_hands(make_vec_env(1024, "call", seed=seed, game=DEFAULT_GAME), _FoldHalfOrCall(seed + 999, DEFAULT_GAME), 2048)
        assert len(r) == 2048
        means.append(r.mean())
    assert abs(np.mean(means) + 0.25) < 0.035, np.mean(means)  # standard error of the 20-seed mean: ~0.009
    assert len(play_hands(make_vec_env(64, "call", seed=0, game=DEFAULT_GAME), _FoldHalfOrCall(1, DEFAULT_GAME), 100)) == 100


def test_play_hands_refuses_an_agent_of_another_tree():
    """Same number of actions, different raise sizes: not the same game (only the action count was compared)."""
    import pytest

    from headsup.env import make_vec_env
    from headsup.game import DEFAULT_GAME, GameConfig

    pot = GameConfig(bet_sizes=(1.0,), mask_redundant=False)
    assert pot.num_actions == DEFAULT_GAME.num_actions
    with pytest.raises(ValueError, match="different game"):
        play_hands(make_vec_env(4, "call", seed=0, game=DEFAULT_GAME), RandomPlayer(seed=0, game=pot), 8)


def test_mask_illegal_normalises_rows_with_tiny_legal_mass():
    """A policy with (almost) all its mass on an illegal action: the legal remainder is renormalised (it was divided
    by max(sum, 1e-12), leaving a row that sums to far less than one)."""
    from headsup.engine import HeadsUpPoker
    from headsup.players import mask_illegal

    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset()
    e.step(1)  # the small blind calls: the big blind has nothing to call, FOLD is illegal
    obs = e.observation()[None]
    probs = mask_illegal(np.array([[1.0, 2e-14, 6e-14, 0.0]]), obs)
    assert probs[0, 0] == 0.0 and probs.sum() == pytest_approx(1.0) and probs[0, 2] == pytest_approx(0.75)
    probs = mask_illegal(np.array([[1.0, 0.0, 0.0, 0.0]]), obs)  # nothing legal left: uniform over the legal actions
    np.testing.assert_allclose(probs[0], [0.0, 1 / 3, 1 / 3, 1 / 3])


def pytest_approx(x):
    import pytest

    return pytest.approx(x, rel=1e-9)


def test_envs_take_and_check_the_network_opponents_game(tmp_path):
    """A network opponent brings its tree (also as numpy weights, where it was ignored: a 3-output FHP net ran in
    the 4-action default game on an unwritten logit); an explicit game of another tree is an error, also for
    trees with the same number of actions."""
    import pytest

    from headsup import native
    from headsup.env import make_vec_env
    from headsup.game import DEFAULT_GAME, FHP, GameConfig
    from headsup.model import BaseModel
    from headsup.players import TorchPolicyPlayer

    fhp_net = BaseModel(features="history", arch="paper", game=FHP)
    pot = GameConfig(bet_sizes=(1.0,))
    pot_net = BaseModel(game=pot)
    if native.available():
        env = make_vec_env(4, fhp_net.numpy_weights(), seed=0)
        assert env.game.tree_dict() == FHP.tree_dict() and env.reset().shape == (4, OBS_DIM)
        assert make_vec_env(4, fhp_net, seed=0).game.tree_dict() == FHP.tree_dict()
        for opponent in (fhp_net, fhp_net.numpy_weights(), pot_net.numpy_weights()):
            with pytest.raises(ValueError, match="different game"):
                make_vec_env(4, opponent, seed=0, game=DEFAULT_GAME)
    player = TorchPolicyPlayer(pot_net, device="cpu", seed=0)
    assert PokerVecEnv(4, player, seed=0).game.tree_dict() == pot.tree_dict()
    with pytest.raises(ValueError, match="different game"):
        PokerVecEnv(4, player, seed=0, game=DEFAULT_GAME)
    with pytest.raises(ValueError, match="different game"):
        PokerVecEnv(4, player, seed=0, raise_cap=4)  # engine overrides must not silently change a network's tree


def test_compare_plays_bot_cells_in_the_tables_game(tmp_path):
    from headsup.compare import head_to_head, table_game
    from headsup.game import FHP
    from headsup.model import BaseModel

    path = tmp_path / "fhp.pth"
    BaseModel(features="history", arch="paper", game=FHP).save(path)
    game = table_game([f"cfr:{path}", "call"], device="cpu")
    assert game.tree_dict() == FHP.tree_dict() and table_game(["call", "random"], device="cpu") is None
    mean, se = head_to_head("raise", "call", 64, num_envs=8, seed=0, device="cpu", game=game)
    assert se > 30  # FHP pots (hundreds of chips), not the 100-chip default game (se <= 12.5 here)


def test_exploit_makes_specs_with_options_absolute(tmp_path, monkeypatch):
    import os

    from headsup.exploit import _absolute_spec

    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "iterates.pt").write_bytes(b"")
    (tmp_path / "x" / "policy.pth").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    root = str(tmp_path / "x")
    assert _absolute_spec("sdcfr:x/iterates.pt") == f"sdcfr:{root}/iterates.pt"
    assert _absolute_spec("sdcfr:x/iterates.pt@exact@k4") == f"sdcfr:{root}/iterates.pt@exact@k4"
    assert _absolute_spec("iterate:x/iterates.pt@t3") == f"iterate:{root}/iterates.pt@t3"
    assert _absolute_spec("search:cfr:x/policy.pth@it100") == f"search:cfr:{root}/policy.pth@it100"
    for spec in ("call", "cfr", "cfr:missing/policy.pth@x", f"cfr:{root}/policy.pth"):
        assert _absolute_spec(spec) == spec
    assert os.getcwd() == str(tmp_path)
