import numpy as np
import pytest
import torch

from headsup import native
from headsup.engine import HeadsUpPoker
from headsup.model import BaseModel, load_model
from headsup.numpy_model import NumpyModel
from headsup.paths import DEFAULT_POLICY_PATH


def random_observations(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    e = HeadsUpPoker(rng=rng)
    rows = []
    while len(rows) < n:
        e.reset()
        while not e.done:
            rows.append(e.observation())
            e.step(rng.integers(4))
    return np.stack(rows[:n])


def test_pretrained_policy_loads_and_beats_calling_station():
    model = load_model(DEFAULT_POLICY_PATH)
    assert sum(p.numel() for p in model.parameters()) == 67844
    from headsup.env import PokerVecEnv, play_hands
    from headsup.players import AlwaysCallPlayer, TorchPolicyPlayer

    env = PokerVecEnv(256, AlwaysCallPlayer(), seed=0)
    r = play_hands(env, TorchPolicyPlayer(model, device="cpu", seed=0), 5000)
    assert r.mean() > 2.0  # ~4.7 chips/hand in practice


def test_numpy_model_matches_torch():
    model = load_model(DEFAULT_POLICY_PATH)
    obs = random_observations()
    with torch.no_grad():
        ref = model(torch.from_numpy(obs)).numpy()
    npm = NumpyModel(model.numpy_weights())
    np.testing.assert_allclose(npm(obs), ref, atol=1e-4)
    np.testing.assert_allclose(np.stack([npm(o) for o in obs[:50]]), ref[:50], atol=1e-4)


def test_fresh_model_is_uniform():
    model = BaseModel()
    obs = torch.from_numpy(random_observations(64))
    assert torch.all(model(obs) == 0)  # zero head -> uniform regret matching / softmax


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_cpp_model_matches_numpy():
    model = load_model(DEFAULT_POLICY_PATH)
    obs = random_observations()
    ref = NumpyModel(model.numpy_weights())(obs)
    cm = native.make_model(model.numpy_weights())
    np.testing.assert_allclose(cm.forward(obs), ref, atol=1e-4)
    np.testing.assert_allclose(cm.forward(obs[3]), ref[3], atol=1e-4)
