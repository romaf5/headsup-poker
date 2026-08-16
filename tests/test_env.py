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
