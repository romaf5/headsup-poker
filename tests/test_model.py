import numpy as np
import pytest
import torch

from headsup import native
from headsup.engine import HeadsUpPoker
from headsup.model import BaseModel, count_parameters, load_model
from headsup.numpy_model import NumpyModel
from headsup.paths import DEFAULT_POLICY_PATH

VARIANTS = [
    dict(),
    dict(features="history"),
    dict(features="both"),
    dict(arch="paper"),
    dict(cards="onehot"),
    dict(features="history", arch="paper", cards="onehot", dim=96, rm_fallback="argmax"),
    dict(features="history", opp_cards=True),
    dict(features="history", arch="paper", opp_cards=True),
    dict(features="both", cards="onehot", opp_cards=True),
]


def _random_model(seed=0, **config):
    torch.manual_seed(seed)
    m = BaseModel(**config)
    with torch.no_grad():  # the head is zero-initialised: give it random weights so outputs differ
        torch.nn.init.normal_(m.action_head.weight, std=0.5)
        torch.nn.init.normal_(m.action_head.bias, std=0.5)
    return m.eval()


def random_observations(n=2000, seed=0, opp_cards=False):
    """Random observation rows; with ``opp_cards`` the history inputs (observation + opponent's cards)."""
    from headsup.model import history_observation

    rng = np.random.default_rng(seed)
    e = HeadsUpPoker(rng=rng)
    rows, opp = [], []
    while len(rows) < n:
        e.reset()
        while not e.done:
            rows.append(e.observation())
            opp.append(e.hands[1 - e.current])
            e.step(rng.integers(4))
    obs = np.stack(rows[:n])
    return history_observation(obs, np.array(opp[:n])) if opp_cards else obs


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


@pytest.mark.parametrize("config", VARIANTS)
def test_variants_torch_numpy_cpp_agree(config):
    model = _random_model(**config)
    obs = random_observations(1500, seed=3, opp_cards=config.get("opp_cards", False))
    with torch.no_grad():
        ref = model(torch.from_numpy(obs)).numpy()
        assert np.abs(ref).max() > 0.1  # non-trivial outputs
        # networks read a prefix of the observation: the exact width gives the same answer
        np.testing.assert_allclose(model(torch.from_numpy(obs[:, : model.obs_dim])).numpy(), ref, atol=1e-6)
    npm = NumpyModel(model.numpy_weights())
    assert npm.obs_dim == model.obs_dim and npm.rm_fallback == model.rm_fallback
    np.testing.assert_allclose(npm(obs), ref, atol=1e-4)
    np.testing.assert_allclose(np.stack([npm(o) for o in obs[:30]]), ref[:30], atol=1e-4)
    if native.available():
        cm = native.make_model(model.numpy_weights())
        assert cm.obs_dim == model.obs_dim and cm.rm_argmax == (model.rm_fallback == "argmax")
        np.testing.assert_allclose(cm.forward(obs), ref, atol=1e-4)
        np.testing.assert_allclose(cm.forward(obs[7]), ref[7], atol=1e-4)


@pytest.mark.parametrize("config", VARIANTS)
def test_config_round_trip(config, tmp_path):
    model = _random_model(**config)
    model.save(tmp_path / "m.pth")
    other = load_model(tmp_path / "m.pth")
    assert other.config == model.config
    obs = torch.from_numpy(random_observations(64, opp_cards=config.get("opp_cards", False)))
    with torch.no_grad():
        np.testing.assert_allclose(other(obs).numpy(), model(obs).numpy())


def test_parameter_counts():
    assert count_parameters(BaseModel()) == 67844
    # DeepCFR paper net (Appendix C, dim 64): 4 card groups x (14 + 5 + 53 embedding rows incl. padding),
    # card1 (256 -> 64), card2/3, bet1 (24 slots x 2 + stack/pot, pot, position = 51 -> 64), bet2,
    # comb1 (128 -> 64), comb2/3, head (64 -> 4)
    paper = BaseModel(features="history", arch="paper")
    d = 64
    expected = 4 * 72 * d + (4 * d * d + d) + 2 * (d * d + d) + (51 * d + d) + (d * d + d) + (2 * d * d + d) + 2 * (d * d + d) + (4 * d + 4)
    assert count_parameters(paper) == expected == 67524


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
