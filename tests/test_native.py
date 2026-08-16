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
