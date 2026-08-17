import numpy as np
import pytest
import torch

from headsup import native
from headsup.deepcfr.memory import ReservoirBuffer
from headsup.deepcfr.traverse import Samples, regret_matching, run_traversals_python
from headsup.model import BaseModel


def test_regret_matching():
    np.testing.assert_allclose(regret_matching(np.array([1.0, 3.0, -2.0, 0.0])), [0.25, 0.75, 0, 0])
    np.testing.assert_allclose(regret_matching(np.array([-1.0, -3.0, -2.0, 0.0])), [0.25] * 4)
    np.testing.assert_allclose(regret_matching(np.array([-1.0, -3.0, -2.0, 0.0]), fold_allowed=False), [0, 1 / 3, 1 / 3, 1 / 3])
    # DeepCFR paper fallback: the highest advantage (among the allowed actions) with probability 1
    np.testing.assert_allclose(regret_matching(np.array([-1.0, -3.0, -2.0, -0.5]), fallback="argmax"), [0, 0, 0, 1])
    np.testing.assert_allclose(regret_matching(np.array([-0.5, -3.0, -2.0, -1.0]), fallback="argmax"), [1, 0, 0, 0])
    np.testing.assert_allclose(regret_matching(np.array([-0.5, -3.0, -2.0, -1.0]), fold_allowed=False, fallback="argmax"), [0, 0, 0, 1])
    np.testing.assert_allclose(regret_matching(np.array([1.0, 3.0, -2.0, 0.0]), fallback="argmax"), [0.25, 0.75, 0, 0])


def test_reservoir_is_uniform_and_round_trips(tmp_path):
    buf = ReservoirBuffer(500, "cpu", obs_dim=31, seed=0)
    for chunk in range(40):
        t = np.arange(chunk * 500, (chunk + 1) * 500, dtype=np.float32)
        buf.add(np.zeros((500, 31), np.float32), t, np.zeros((500, 4), np.float32))
    assert len(buf) == 500 and buf.seen == 20000
    ts = buf.t.numpy()
    assert 8000 < ts.mean() < 12000  # uniform over [0, 20000)
    buf.save(tmp_path / "buf.pt")
    other = ReservoirBuffer(500, "cpu", obs_dim=31).load(tmp_path / "buf.pt")
    assert other.seen == buf.seen and torch.equal(other.t, buf.t)
    obs, t, target = other.sample(16)
    assert obs.shape == (16, 31) and t.shape == (16,) and target.shape == (16, 4)


def test_reservoir_host_storage_samples_to_device():
    """Memories on the CPU sampled to another device (here also cpu -> exercises the staging path)."""
    from headsup.engine import OBS_DIM, HeadsUpPoker

    rng = np.random.default_rng(1)
    e = HeadsUpPoker(rng=rng)
    rows = []
    while len(rows) < 200:
        e.reset()
        while not e.done:
            rows.append(e.observation())
            e.step(rng.integers(4))
    obs = np.stack(rows[:200])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    buf = ReservoirBuffer(1000, "cpu", obs_dim=OBS_DIM, seed=0, sample_device=dev)
    buf.add(obs, np.arange(200, dtype=np.float32), rng.random((200, 4), dtype=np.float32))
    seen = 0
    for got, t, target in buf.prefetch(64, 7):
        assert str(got.device).startswith(dev) and got.shape == (64, OBS_DIM)
        rows_idx = t.long().cpu().numpy()
        np.testing.assert_array_equal(got.cpu().numpy(), obs[rows_idx])
        np.testing.assert_array_equal(target.cpu().numpy(), buf.target[rows_idx].numpy())
        seen += 1
    assert seen == 7


def test_reservoir_stores_observations_exactly(tmp_path):
    from headsup.engine import OBS_DIM, HeadsUpPoker

    rng = np.random.default_rng(0)
    e = HeadsUpPoker(rng=rng)
    rows = []
    while len(rows) < 300:
        e.reset()
        while not e.done:
            rows.append(e.observation())
            e.step(rng.integers(4))
    obs = np.stack(rows[:300])
    for obs_dim in (31, OBS_DIM):
        buf = ReservoirBuffer(1000, "cpu", obs_dim=obs_dim, seed=0)
        buf.add(obs, np.arange(300, dtype=np.float32), rng.random((300, 4), dtype=np.float32))
        np.testing.assert_array_equal(buf.obs.numpy(), obs[:, :obs_dim])  # uint8 / float32 split is lossless
        got, t, _ = buf.sample(50)
        np.testing.assert_array_equal(got.numpy(), obs[t.long().numpy(), :obs_dim])
        buf.save(tmp_path / "buf.pt")
        with pytest.raises(ValueError):  # a buffer of another width refuses the file
            ReservoirBuffer(1000, "cpu", obs_dim=OBS_DIM if obs_dim == 31 else 31).load(tmp_path / "buf.pt")


VARIANTS = [
    dict(),
    dict(features="history"),
    dict(features="both", arch="paper", cards="onehot", dim=32),
    dict(features="history", arch="paper", rm_fallback="argmax"),
    dict(cards="onehot", rm_fallback="argmax"),
]


def _peaked(bias, **config):
    """Zero head weights + a bias: the advantages equal ``bias`` everywhere, so the strategy is
    deterministic (a one-hot from regret matching, or from the argmax fallback when all entries
    are negative) and C++ / Python traversals take identical paths despite different RNG streams."""
    m = BaseModel(**config)
    with torch.no_grad():
        m.action_head.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return m.numpy_weights()


def test_python_traversal_shapes():
    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, strat, nodes = run_traversals_python(w, 0, 20, 1.0, seed=0)
    assert isinstance(adv, Samples) and adv.obs.shape[1] == 31 and adv.target.shape[1] == 4
    hw = [BaseModel(features="history").numpy_weights() for _ in range(2)]
    hadv, hstrat, _ = run_traversals_python(hw, 0, 5, 1.0, seed=0)
    assert hadv.obs.shape[1] == 79 and hstrat.obs.shape[1] == 79
    assert nodes >= len(adv) + len(strat) > 0
    assert np.all(adv.t == 1.0)
    # advantages are centred by the current strategy: uniform for fresh nets where fold is
    # allowed, uniform over the 3 remaining actions (fold == check) where nothing is to call
    fold_ok = adv.obs[:, 23] > 0
    np.testing.assert_allclose(adv.target[fold_ok].mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(adv.target[~fold_ok][:, 1:].mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(adv.target[~fold_ok][:, 0], adv.target[~fold_ok][:, 1], atol=1e-6)
    np.testing.assert_allclose(strat.target.sum(axis=1), 1, atol=1e-5)
    assert np.all(strat.target[strat.obs[:, 23] <= 0, 0] == 0)  # never fold for free


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
@pytest.mark.parametrize("config", VARIANTS)
def test_cpp_traversal_matches_python_reference(config):
    cpp = native.module()
    rng = np.random.default_rng(0)
    n = 60
    decks = np.stack([rng.permutation(52)[:9] for _ in range(n)]).astype(np.int32)
    if config.get("rm_fallback") == "argmax":  # all negative: every decision goes through the fallback
        w0, w1 = _peaked([-3, -1, -2, -4], **config), _peaked([-2, -3, -1, -4], **config)
    else:
        w0, w1 = _peaked([0, 5, 0, 0], **config), _peaked([0, 0, 5, 0], **config)
    for traverser in (0, 1):
        pa, ps, pn = run_traversals_python([w0, w1], traverser, n, 3.0, 1, None, decks)
        out = cpp.run_traversals(native.make_model(w0), native.make_model(w1), traverser, n, 3.0, 1, native.engine_config(), decks)
        assert pn == out[6] and len(pa) == len(out[1]) and len(ps) == len(out[4])
        assert pa.obs.shape[1] == out[0].shape[1] == (31 if config.get("features", "aggregated") == "aggregated" else 79)
        np.testing.assert_array_equal(pa.obs, out[0])
        np.testing.assert_allclose(pa.target, out[2], atol=1e-4)
        np.testing.assert_allclose(ps.target, out[5], atol=1e-5)


def test_train_advantage_and_policy_smoke():
    from headsup.deepcfr.train import train_advantage_net, train_policy_net

    buf = ReservoirBuffer(10000, "cpu", obs_dim=31, seed=0)
    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, strat, _ = run_traversals_python(w, 0, 100, 1.0, seed=0)
    buf.add(adv.obs, adv.t, adv.target)
    net, final_loss = train_advantage_net(buf, "cpu", steps=5, batch_size=64, compile=False)
    assert isinstance(net, BaseModel) and np.isfinite(final_loss)
    sbuf = ReservoirBuffer(10000, "cpu", obs_dim=31, seed=0)
    sbuf.add(strat.obs, strat.t, strat.target)
    policy = train_policy_net(sbuf, "cpu", epochs=1, batch_size=64, compile=False, progress=False)
    assert isinstance(policy, BaseModel)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_cpp_dream_sampler_matches_python_reference():
    """DREAM outcome sampling with eps = 0 and one-hot strategies is deterministic given the deal:
    the C++ sampler and the Python reference produce the same advantage / baseline samples."""
    from headsup.deepcfr.traverse import run_dream_python
    from headsup.model import OBS_DIM_WITH_OPP

    cpp = native.module()
    rng = np.random.default_rng(1)
    n = 80
    decks = np.stack([rng.permutation(52)[:9] for _ in range(n)]).astype(np.int32)
    w0, w1 = _peaked([0, 5, 0, 0], features="history"), _peaked([0, 0, 5, 0], features="history")
    torch.manual_seed(0)
    baseline = BaseModel(features="history", opp_cards=True)
    with torch.no_grad():
        torch.nn.init.normal_(baseline.action_head.weight, std=0.5)
    bw = baseline.numpy_weights()
    for traverser in (0, 1):
        pa, pv, pn = run_dream_python([w0, w1], bw, traverser, n, 3.0, 0.0, 1, None, decks)
        out = cpp.run_dream(native.make_model(w0), native.make_model(w1), native.make_model(bw), traverser, n, 3.0, 0.0, 1,
                            native.engine_config(), decks)
        assert pn == out[6] and len(pa) == len(out[1]) > 0 and len(pv) == len(out[4]) > 0
        assert out[3].shape[1] == OBS_DIM_WITH_OPP
        np.testing.assert_array_equal(pa.obs, out[0])
        np.testing.assert_allclose(pa.t, np.ravel(out[1]), rtol=1e-5)  # t / own sample reach (= t: xi one-hot along the path)
        np.testing.assert_allclose(pa.target, out[2], atol=1e-3)
        np.testing.assert_array_equal(pv.obs, out[3])
        np.testing.assert_array_equal(pv.t, np.ravel(out[4]))  # the action taken at each history
        np.testing.assert_allclose(pv.target, out[5], atol=1e-3)
    # the runner wraps the same call
    from headsup.deepcfr.traverse import TraversalRunner

    with TraversalRunner(2, backend="cpp") as runner:
        adv, val, nodes = runner.collect_dream([w0, w1], bw, 0, 40, 2.0, 0.5, seed=3)
        assert nodes > 0 and len(adv) > 0 and val.obs.shape[1] == OBS_DIM_WITH_OPP
        assert np.all(adv.t >= 2.0)  # t / own sample reach >= t


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_cpp_escher_samplers():
    """ESCHER value rows carry player 0's return of the trajectory; regret rows are q - sigma.q
    of the value net (checked by recomputing them from the returned history rows)."""
    from headsup.deepcfr.traverse import TraversalRunner
    from headsup.model import OBS_DIM_WITH_OPP
    from headsup.numpy_model import NumpyModel

    w0, w1 = _peaked([0, 5, 0, 0], features="history"), _peaked([0, 0, 5, 0], features="history")
    torch.manual_seed(0)
    vnet = BaseModel(features="history", opp_cards=True)
    with torch.no_grad():
        torch.nn.init.normal_(vnet.action_head.weight, std=0.5)
    vw = vnet.numpy_weights()
    with TraversalRunner(2, backend="cpp") as runner:
        val, nodes = runner.collect_escher_values([w0, w1], 50, seed=0)
        assert val.obs.shape[1] == OBS_DIM_WITH_OPP and len(val) >= 50 and nodes == len(val)
        picked = val.target[np.arange(len(val)), val.t.astype(int)]  # u_0 of the trajectory in the taken action's slot
        assert np.all(np.abs(picked) <= 100.0) and np.all(val.target.sum(1) == picked)
        adv, strat, hist, nodes = runner.collect_escher_regrets([w0, w1], vw, 1, 30, 4.0, seed=0)
        assert len(adv) == len(strat) == len(hist) > 0 and np.all(adv.t == 4.0)
        q = -NumpyModel(vw)(hist.obs)  # player 1's values
        legal = adv.target != 0  # (illegal / duplicate actions have target 0; so may a legal one, rarely)
        v = hist.target[:, 0]
        expected = np.where(legal, q - v[:, None], 0.0)
        np.testing.assert_allclose(adv.target[legal], expected[legal], atol=1e-3)
        np.testing.assert_allclose(strat.target.sum(1), 1.0, atol=1e-5)
