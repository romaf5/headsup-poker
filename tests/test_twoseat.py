"""The two-seat vectorised env (both seats driven from Python): the Python twin against the one-seat env, the C++
env against the twin."""

import numpy as np
import pytest

from headsup import native
from headsup.engine import OBS_DIM
from headsup.env import PokerVecEnv
from headsup.game import DEFAULT_GAME, GameConfig
from headsup.twoseat import NativeTwoSeatVecEnv, TwoSeatVecEnv, make_two_seat_env

needs_cpp = pytest.mark.skipif(not native.available(), reason="C++ extension not built")
POT_GAME = GameConfig(bet_sizes=(0.5, 1.0), mask_redundant=True)


def _decks(rng, n):
    return np.stack([rng.permutation(52)[:9] for _ in range(n)])


class _ObsPlayer:
    """A deterministic stateless player: the action is a fixed function of the observation."""

    def __init__(self, game, seed):
        self.game = game
        self.w = np.random.default_rng(seed).uniform(1.0, 2.0, OBS_DIM)

    def __call__(self, obs, ids=None):
        key = np.floor(np.asarray(obs, dtype=np.float64) @ self.w * 977.0).astype(np.int64)
        return (key % (self.game.num_actions + 2)).clip(max=self.game.num_actions - 1)  # the all-in more often: short hands


class _FoldHalfRaiseSome(_ObsPlayer):
    """Folds half of its first small-blind decisions, otherwise raises a quarter of the time and calls the rest:
    against a calling station that is -0.25 chips/hand with a small variance."""

    def __call__(self, obs, ids=None):
        key = np.floor(np.asarray(obs, dtype=np.float64) @ self.w * 977.0).astype(np.int64)
        first_sb = (obs[:, 21] == 0) & (obs[:, 22] == 0) & (obs[:, 32] == 0)
        return np.where(first_sb & (key % 2 == 0), 0, np.where(key % 4 == 1, 2, 1))


def test_twin_turn_order_rewards_and_waiting():
    env = TwoSeatVecEnv(4, seed=0)
    obs, seat = env.reset()
    assert obs.shape == (4, OBS_DIM) and obs.dtype == np.float32 and seat.tolist() == [0, 0, 0, 0]
    assert (obs[:, 22] == 0).all()  # the small blind's observation
    first = obs.copy()
    obs, seat, rewards, dones = env.step([0, 1, -1, 2])  # fold / limp / wait / raise
    assert dones.tolist() == [True, False, False, False] and rewards.shape == (4, 2) and rewards.dtype == np.float32
    assert rewards[0].tolist() == [-1.0, 1.0] and not rewards[1:].any()
    assert seat.tolist() == [0, 1, 0, 1]  # a new hand at table 0; the big blind at 1 and 3; table 2 waited
    np.testing.assert_array_equal(obs[2], first[2])
    assert (obs[[1, 3], 22] == 1).all() and obs[0, 22] == 0
    assert env.hands_completed == 1
    obs, seat, rewards, dones = env.step([-1, 1, -1, 0])  # the big blind checks (flop: it acts first) / folds to the raise
    assert dones.tolist() == [False, False, False, True] and rewards[3].tolist() == [2.0, -2.0]
    assert seat.tolist() == [0, 1, 0, 0] and obs[1, 21] == 1
    with pytest.raises(ValueError, match="[Ii]nvalid action"):
        env.step([1, 4, 1, 1])
    with pytest.raises(ValueError):
        env.step([1, 1, 1])


def test_twin_takes_decks_for_the_next_deal():
    env = TwoSeatVecEnv(2, seed=0)
    decks = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8], [51, 50, 49, 48, 47, 46, 45, 44, 43]])
    obs, _ = env.reset(decks)
    assert obs[0, 2] == 1 and obs[0, 5] == 2 and obs[1, 2] == 51 and obs[1, 5] == 52  # seat 0's hole cards (card + 1), sorted
    nxt = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18], [20, 21, 22, 23, 24, 25, 26, 27, 28]])
    obs, seat, _, dones = env.step([0, 1], nxt)  # only table 0 is dealt again
    assert dones.tolist() == [True, False] and obs[0, 2] == 11 and obs[0, 5] == 12
    assert seat[1] == 1 and obs[1, 2] == 49 and obs[1, 5] == 50  # seat 1 of the first deal


def test_twin_plays_the_hands_of_the_one_seat_env():
    """Same seed -> same cards; with deterministic players every table's hands must give the agent the rewards the
    one-seat env reports (its seat alternates between hands; an opponent's open-fold is paid on the next step)."""
    n, hands = 8, 30
    agent, opponent = _ObsPlayer(DEFAULT_GAME, 1), _ObsPlayer(DEFAULT_GAME, 2)
    one = PokerVecEnv(n, opponent, seed=11)
    expected = [[] for _ in range(n)]
    obs = one.reset()
    while min(len(x) for x in expected) < hands:
        obs, r, d, _ = one.step(agent(obs))
        for i in np.flatnonzero(d):
            expected[i].append(float(r[i]))
    two = TwoSeatVecEnv(n, seed=11)
    agent_seat = np.arange(n) % 2  # PokerVecEnv: table 0 starts in seat 0, table 1 in seat 1, ...
    got = [[] for _ in range(n)]
    obs, seat = two.reset()
    while min(len(x) for x in got) < hands:
        mine = seat == agent_seat
        actions = np.where(mine, agent(obs), opponent(obs))
        obs, seat, r, d = two.step(actions)
        for i in np.flatnonzero(d):
            got[i].append(float(r[i, agent_seat[i]]))
            agent_seat[i] ^= 1
    for i in range(n):
        assert got[i][:hands] == expected[i][:hands]
    seen = {abs(x) for row in got for x in row}
    assert {1.0, 2.0, 100.0} <= seen and len(seen) >= 5  # folds, showdowns and all-ins all occurred


@needs_cpp
@pytest.mark.parametrize("game", [DEFAULT_GAME, POT_GAME], ids=["default", "pot"])
@pytest.mark.parametrize("allin_ev", [False, True], ids=["dealt", "allin_ev"])
def test_cpp_env_matches_the_python_twin(game, allin_ev):
    rng = np.random.default_rng(3)
    n = 12
    twin, cpp = TwoSeatVecEnv(n, seed=1, game=game), NativeTwoSeatVecEnv(n, seed=2, game=game)
    for env in (twin, cpp):
        env.allin_ev, env.ev_samples, env.ev_seed = allin_ev, 60, 987654321
    decks = _decks(rng, n)
    for x, y in zip(twin.reset(decks), cpp.reset(decks)):
        np.testing.assert_array_equal(x, y)
    seen_fractional = False
    for _ in range(350):
        actions = rng.integers(0, game.num_actions, n)
        actions[rng.random(n) < 0.1] = -1  # waiting tables
        decks = _decks(rng, n)
        a, b = twin.step(actions, decks), cpp.step(actions, decks)
        for x, y in zip(a, b):
            assert x.dtype == y.dtype and x.shape == y.shape
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a[2][:, 0], -a[2][:, 1])  # zero-sum
        assert not a[2][~a[3]].any()
        seen_fractional |= bool((a[2] != np.rint(a[2])).any())
    assert twin.hands_completed == cpp.hands_completed > 300
    assert seen_fractional == allin_ev  # expectations over runouts are not whole chips


@needs_cpp
def test_cpp_env_deals_by_itself_and_matches_the_one_seat_env_statistically():
    """Its own generator: valid deals, and the same chips/hand as the one-seat C++ env for the same pair of players."""
    from headsup.env import NativeVecEnv, play_hands

    agent = _FoldHalfRaiseSome(DEFAULT_GAME, 5)
    one = play_hands(NativeVecEnv(512, "call", seed=0), agent, 30_000)
    env = make_two_seat_env(512, seed=0)
    assert isinstance(env, NativeTwoSeatVecEnv) and env.game == DEFAULT_GAME and env.num_envs == 512
    agent_seat = np.arange(512) % 2
    obs, seat = env.reset()
    two = []
    while len(two) < 30_000:
        cards = obs[:, [2, 5, 8, 11, 14, 17, 20]]
        assert all(len(set(row[row > 0])) == (row > 0).sum() for row in cards[:16])  # distinct cards
        obs, seat, r, d = env.step(np.where(seat == agent_seat, agent(obs), 1))
        two.extend(r[d, agent_seat[d]].tolist())
        agent_seat[d] ^= 1
    two = np.array(two[:30_000])
    se = np.hypot(one.std(), two.std()) / np.sqrt(30_000)
    assert abs(one.mean() - two.mean()) < 4 * se and abs(one.std() - two.std()) < 0.1 * one.std()
    assert abs(one.mean()) > 4 * se  # the pair is not a coin flip: the comparison means something


@needs_cpp
def test_cpp_env_rejects_bad_input():
    env = NativeTwoSeatVecEnv(3, seed=0)
    env.reset()
    with pytest.raises(ValueError, match="[Ii]nvalid action"):
        env.step([1, 4, 1])
    with pytest.raises(ValueError):
        env.step([1, 1])
    with pytest.raises(ValueError):
        env.reset(np.zeros((3, 9), dtype=np.int64))  # not nine distinct cards
    with pytest.raises(ValueError):
        env.reset(np.arange(18).reshape(2, 9))  # one deck per table
    obs, seat, r, d = env.step([-1, -1, -1])  # everybody waits: nothing happens
    assert not d.any() and not r.any() and seat.tolist() == [0, 0, 0]


def test_make_two_seat_env_backends():
    env = make_two_seat_env(4, seed=0, backend="python", game=POT_GAME)
    assert isinstance(env, TwoSeatVecEnv) and env.game == POT_GAME
    obs, seat = env.reset()
    assert obs.shape == (4, OBS_DIM)
