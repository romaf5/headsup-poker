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


def test_reservoir_is_uniform_and_round_trips(tmp_path):
    buf = ReservoirBuffer(500, "cpu", seed=0)
    for chunk in range(40):
        t = np.arange(chunk * 500, (chunk + 1) * 500, dtype=np.float32)
        buf.add(np.zeros((500, 31), np.float32), t, np.zeros((500, 4), np.float32))
    assert len(buf) == 500 and buf.seen == 20000
    ts = buf.t.numpy()
    assert 8000 < ts.mean() < 12000  # uniform over [0, 20000)
    buf.save(tmp_path / "buf.pt")
    other = ReservoirBuffer(500, "cpu").load(tmp_path / "buf.pt")
    assert other.seen == buf.seen and torch.equal(other.t, buf.t)
    obs, t, target = other.sample(16)
    assert obs.shape == (16, 31) and t.shape == (16,) and target.shape == (16, 4)


def _peaked(bias):
    m = BaseModel()
    with torch.no_grad():
        m.action_head.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return m.numpy_weights()


def test_python_traversal_shapes():
    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, strat, nodes = run_traversals_python(w, 0, 20, 1.0, seed=0)
    assert isinstance(adv, Samples) and adv.obs.shape[1] == 31 and adv.target.shape[1] == 4
    assert nodes >= len(adv) + len(strat) > 0
    assert np.all(adv.t == 1.0)
    # advantages are centred by the current strategy (uniform for fresh nets)
    np.testing.assert_allclose(adv.target.mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(strat.target.sum(axis=1), 1, atol=1e-5)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_cpp_traversal_matches_python_reference():
    cpp = native.module()
    rng = np.random.default_rng(0)
    n = 100
    decks = np.stack([rng.permutation(52)[:9] for _ in range(n)]).astype(np.int32)
    w0, w1 = _peaked([0, 5, 0, 0]), _peaked([0, 0, 5, 0])  # deterministic strategies
    for traverser in (0, 1):
        pa, ps, pn = run_traversals_python([w0, w1], traverser, n, 3.0, 1, None, decks)
        out = cpp.run_traversals(native.make_model(w0), native.make_model(w1), traverser, n, 3.0, 1, native.engine_config(), decks)
        assert pn == out[6] and len(pa) == len(out[1]) and len(ps) == len(out[4])
        np.testing.assert_array_equal(pa.obs, out[0])
        np.testing.assert_allclose(pa.target, out[2], atol=1e-4)
        np.testing.assert_allclose(ps.target, out[5], atol=1e-5)


def test_train_advantage_and_policy_smoke():
    from headsup.deepcfr.train import train_advantage_net, train_policy_net

    buf = ReservoirBuffer(10000, "cpu", seed=0)
    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, strat, _ = run_traversals_python(w, 0, 100, 1.0, seed=0)
    buf.add(adv.obs, adv.t, adv.target)
    net, final_loss = train_advantage_net(buf, "cpu", steps=5, batch_size=64, compile=False)
    assert isinstance(net, BaseModel) and np.isfinite(final_loss)
    sbuf = ReservoirBuffer(10000, "cpu", seed=0)
    sbuf.add(strat.obs, strat.t, strat.target)
    policy = train_policy_net(sbuf, "cpu", epochs=1, batch_size=64, compile=False, progress=False)
    assert isinstance(policy, BaseModel)
