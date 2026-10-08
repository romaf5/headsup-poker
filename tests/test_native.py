import numpy as np
import pytest

from headsup import native
from headsup.cards import hand_strength
from headsup.engine import HeadsUpPoker

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")


def test_eval7_matches_treys():
    cpp = native.module()
    rng = np.random.default_rng(0)
    for _ in range(3000):
        cards = rng.permutation(52)[:7].tolist()
        assert cpp.eval7(cards) == hand_strength(cards[:2], cards[2:])


def test_cpp_engine_matches_python_engine():
    cpp = native.module()
    rng = np.random.default_rng(2)
    py = HeadsUpPoker(rng=rng)
    ce = cpp.Engine()
    for _ in range(3000):
        perm = rng.permutation(52)
        o_py = py.reset(deck=perm)
        ce.reset(perm.tolist())
        while True:
            assert py.current == ce.current
            np.testing.assert_array_equal(o_py, ce.observation())
            a = int(rng.choice(4, p=[0.1, 0.4, 0.35, 0.15]))
            o_py, r_py, d_py, _ = py.step(a)
            d_c = ce.step(a)
            assert d_py == d_c
            if d_py:
                assert list(r_py) == list(ce.rewards) and py.folded == ce.folded
                break


def test_native_vecenv_rewards_are_consistent():
    from headsup.env import NativeVecEnv, PokerVecEnv, play_hands
    from headsup.players import AlwaysCallPlayer, RandomPlayer

    # calling station vs calling station: low variance, so means/stds are tightly comparable
    r_native = play_hands(NativeVecEnv(512, "call", seed=0), AlwaysCallPlayer(), 40000)
    r_py = play_hands(PokerVecEnv(512, AlwaysCallPlayer(), seed=0), AlwaysCallPlayer(), 40000)
    assert abs(r_native.mean() - r_py.mean()) < 0.15
    assert abs(r_native.std() - r_py.std()) < 0.5
    # random opponent: fold is never taken when nothing is to call (both implementations)
    r_native = play_hands(NativeVecEnv(512, "random", seed=0), AlwaysCallPlayer(), 20000)
    r_py = play_hands(PokerVecEnv(512, RandomPlayer(seed=0), seed=0), AlwaysCallPlayer(), 20000)
    assert abs(r_native.mean() - r_py.mean()) < 3 * (r_native.std() + r_py.std()) / np.sqrt(20000)


def test_sampling_never_returns_a_zero_probability_action():
    """float32 partial sums can stop short of 1: the remainder belongs to the last action WITH probability (the
    last index was returned - in a limit game the illegal raise at the cap, ~1e-8 per draw; DREAM then stored
    advantage rows with weight t / 1e-12 and the sampled subgame solver read child -1)."""
    cpp = native.module()
    p = np.array([0.5, 0.49999994, 0.0], dtype=np.float32)
    u = np.float32(0.99999994)
    assert np.float32(p[0] + p[1]) <= u < 1.0  # the draw lies beyond the partial sums
    assert cpp.sample_index(p, float(u)) == 1
    assert cpp.sample_index(np.array([0.0, 1.0, 0.0], np.float32), 0.0) == 1
    assert cpp.sample_index(np.array([0.25, 0.0, 0.75], np.float32), 0.25) == 2
    assert cpp.sample_index(np.array([0.25, 0.0, 0.75], np.float32), 0.2) == 0


def test_native_fixed_bots_play_the_twin_of_a_masked_action():
    """The native raise bot played action 2 even where the game masks it; the Python bot plays its twin - the same
    chips, but another raise counter in what the agent then observes, and replay-based players left their tree."""
    from headsup.env import make_vec_env
    from headsup.game import GameConfig

    game = GameConfig(bet_sizes=(0.5, 1.0), mask_redundant=True)
    seen = {}
    for backend in ("cpp", "python"):
        env = make_vec_env(64, "raise", seed=0, game=game, backend=backend)
        obs, rng, counters = env.reset(), np.random.default_rng(0), set()
        for _ in range(300):
            facing_allin = (obs[:, 28] > 0) & (np.abs(obs[:, 23] - obs[:, 28]) < 1e-6)  # to call == the whole remaining stack
            counters |= set(obs[facing_allin, 79].astype(int).tolist())
            obs, _, _, _ = env.step(np.where(rng.random(64) < 0.5, 2, 1))
        seen[backend] = counters
    assert seen["cpp"] == seen["python"] and seen["cpp"]


def test_vec_env_rejects_out_of_range_actions():
    from headsup.env import make_vec_env

    env = make_vec_env(4, "call", seed=0, backend="cpp")
    env.reset()
    for bad in (7, -1):
        with pytest.raises(ValueError, match="[Ii]nvalid action"):
            env.step(np.array([1, bad, 1, 1]))


def test_solvers_map_masked_actions_to_their_twins_and_bound_the_buckets():
    from headsup.game import GameConfig

    cpp = native.module()
    game = GameConfig(bet_sizes=(0.5, 1.0), mask_redundant=True)
    e = cpp.Engine(native.engine_config(game=game))
    e.reset(list(range(9)))
    while e.stage == 0:  # to the flop by calling
        e.step(1)
    sv = cpp.VectorSolver()
    sv.build(e, 50)
    legal = np.asarray(e.legal_mask())
    assert not legal.all()  # the flop root of this game has a masked action (fold: nothing to call)
    for a in np.flatnonzero(~legal):
        assert sv.child(0, int(a)) >= 0  # executed as its twin, as the engine does
    with pytest.raises((RuntimeError, ValueError), match="buckets"):
        cpp.VectorSolver().build(e, 3000)  # more buckets than hands overflowed the per-level scratch


def test_engine_config_is_validated():
    cpp = native.module()
    cfg = cpp.EngineConfig()
    cfg.bet_sizes = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # 11 actions > MAX_ACTIONS
    with pytest.raises((RuntimeError, ValueError), match="actions"):
        cpp.Engine(cfg)
    cfg = cpp.EngineConfig()
    cfg.limit, cfg.num_rounds, cfg.has_all_in, cfg.bet_sizes = [100], 2, False, [-2.0]  # one increment for two rounds
    with pytest.raises((RuntimeError, ValueError), match="round"):
        cpp.Engine(cfg)
