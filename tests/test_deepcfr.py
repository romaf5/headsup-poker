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
        np.testing.assert_array_equal(pa.legal, out[7])
        np.testing.assert_array_equal(out[7][:, 0], pa.obs[:, 23] > 0)  # fold is legal exactly when facing a bet


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
        np.testing.assert_array_equal(pa.legal, out[7])
        assert not np.any(out[2][~out[7]])  # DREAM: illegal actions' regret targets are 0
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
        val, nodes = runner.collect_escher_values([w0, w1], 50, seed=0, value_epsilon=0.0)  # on-policy: plain returns
        assert val.obs.shape[1] == OBS_DIM_WITH_OPP and len(val) >= 50 and nodes == len(val)
        picked = val.target[np.arange(len(val)), val.t.astype(int)]  # u_0 of the trajectory in the taken action's slot
        assert np.all(np.abs(picked) <= 100.0) and np.all(val.target.sum(1) == picked)
        assert set(np.unique(val.obs[:, 22])) == {0.0, 1.0}  # the history input says whose turn it is
        # with exploration, returns are importance-weighted by sigma / xi of the later actions: peaked
        # strategies make off-policy continuations worth ~0, on-policy ones ~u_0 / 0.99^k
        val_x, _ = runner.collect_escher_values([w0, w1], 400, seed=1, value_epsilon=0.3)
        picked_x = val_x.target[np.arange(len(val_x)), val_x.t.astype(int)]
        assert np.all(np.abs(picked_x) <= 100.0 / 0.7**3) and (picked_x == 0).sum() < len(picked_x)
        adv, strat, hist, nodes = runner.collect_escher_regrets([w0, w1], vw, 1, 30, 4.0, seed=0)
        assert len(adv) == len(hist) > 0 and len(strat) > 0 and np.all(adv.t == 4.0) and np.all(strat.t == 4.0)
        # regrets at the update player's (seat 1) infosets, average-policy rows at the opponent's (seat 0)
        assert np.all(adv.obs[:, 22] == 1.0) and np.all(strat.obs[:, 22] == 0.0)
        q = -NumpyModel(vw)(hist.obs)  # player 1's values
        legal = adv.legal
        assert legal.shape == adv.target.shape and not np.any(adv.target[~legal])
        v = hist.target[:, 0]
        expected = np.where(legal, q - v[:, None], 0.0)
        np.testing.assert_allclose(adv.target[legal], expected[legal], atol=1e-3)
        np.testing.assert_allclose(strat.target.sum(1), 1.0, atol=1e-5)


def test_advantage_fit_scales_large_targets():
    """Regrets of hundreds of chips (FHP) are fitted in scaled units and the output layer is scaled
    back: the net predicts chips, while a raw-chip fit cannot grow its zero-initialised head in time."""
    from headsup.deepcfr.train import train_advantage_net

    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, _, _ = run_traversals_python(w, 0, 100, 1.0, seed=0)
    target = np.tile(np.array([300.0, -300.0, 0.0, 0.0], np.float32), (len(adv), 1))
    buf = ReservoirBuffer(10000, "cpu", obs_dim=31, seed=0)
    buf.add(adv.obs, np.ones(len(adv), np.float32), target)
    obs = torch.as_tensor(adv.obs[:256, :31])
    torch.manual_seed(0)
    scaled, _ = train_advantage_net(buf, "cpu", steps=300, batch_size=64, compile=False)
    torch.manual_seed(0)
    raw, _ = train_advantage_net(buf, "cpu", steps=300, batch_size=64, compile=False, target_scale=None)
    with torch.no_grad():
        np.testing.assert_allclose(scaled(obs).mean(0).numpy(), [300.0, -300.0, 0.0, 0.0], atol=30.0)
        assert raw(obs)[:, 0].mean().item() < 150.0


def _bits(i, width=4):
    return (np.asarray(i)[:, None] >> np.arange(width)) & 1


def test_reservoir_keeps_legal_masks_with_their_rows(tmp_path):
    """Legal masks follow their samples through reservoir replacement, sampling (also via host staging) and save / load."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    buf = ReservoirBuffer(300, "cpu", obs_dim=31, legal_dim=4, seed=0, sample_device=dev)
    for chunk in range(5):
        i = np.arange(chunk * 200, (chunk + 1) * 200)
        buf.add(np.zeros((200, 31), np.float32), i.astype(np.float32), np.zeros((200, 4), np.float32), _bits(i % 16))
    assert len(buf) == 300 and buf.seen == 1000 and buf.t.max() >= 300  # rows were replaced
    np.testing.assert_array_equal(buf.legal.numpy(), _bits(buf.t.long().numpy() % 16))
    for obs, t, target, legal in buf.prefetch(64, 3, with_legal=True):
        np.testing.assert_array_equal(legal.cpu().numpy(), _bits(t.long().cpu().numpy() % 16))
    assert len(buf.sample(8)) == 3  # without with_legal: the usual triple
    buf.save(tmp_path / "buf.pt")
    other = ReservoirBuffer(300, "cpu", obs_dim=31, legal_dim=4).load(tmp_path / "buf.pt")
    assert torch.equal(other.legal, buf.legal)
    plain = ReservoirBuffer(300, "cpu", obs_dim=31, seed=0)
    plain.add(np.zeros((50, 31), np.float32), np.ones(50, np.float32), np.zeros((50, 4), np.float32))
    plain.save(tmp_path / "plain.pt")
    old = ReservoirBuffer(300, "cpu", obs_dim=31, legal_dim=4).load(tmp_path / "plain.pt")  # saved without masks: all legal
    assert old.legal[:50].all()
    with pytest.raises(ValueError):
        plain.sample(4, with_legal=True)


def test_masked_advantage_fit_ignores_illegal_targets():
    """With the masked loss the targets of illegal actions do not influence the fit at all."""
    from headsup.deepcfr.train import train_advantage_net

    rng = np.random.default_rng(0)
    obs = rng.random((500, 31), dtype=np.float32)
    obs[:, :23] = rng.integers(0, 3, (500, 23))
    t = rng.integers(1, 10, 500).astype(np.float32)
    target = rng.normal(size=(500, 4)).astype(np.float32)
    legal = rng.random((500, 4)) < 0.7
    legal[:, 1] = True
    garbage = np.where(legal, target, 1000.0).astype(np.float32)
    fits = {}
    for name, tg in (("clean", target), ("garbage", garbage)):
        for masked in (True, False):
            buf = ReservoirBuffer(1000, "cpu", obs_dim=31, legal_dim=4, seed=0)
            buf.add(obs, t, tg, legal)
            torch.manual_seed(0)
            net, _ = train_advantage_net(buf, "cpu", steps=20, batch_size=64, compile=False, masked=masked)
            fits[name, masked] = net(torch.as_tensor(obs)).detach()
    torch.testing.assert_close(fits["clean", True], fits["garbage", True])
    assert not torch.allclose(fits["clean", False], fits["garbage", False])


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_trainer_masked_loss_fhp_stores_masks_and_resumes(tmp_path):
    from headsup.deepcfr.train import main as train_main

    common = ["--workers", "2", "--device", "cpu", "--no-compile", "--traversals", "30", "--value-steps", "3", "--batch-size", "64",
              "--eval-hands", "0", "--eval-every", "0", "--policy-eval-every", "0", "--lbr-every", "0", "--lbr-final-hands", "0",
              "--no-tensorboard", "--checkpoint-every", "1", "--out", str(tmp_path)]
    train_main(["--game", "fhp", "--algo", "sdcfr", "--masked-loss", "--features", "history", "--net", "paper",
                "--adv-capacity", "20000", "--iterations", "2"] + common)
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["args"]["masked_loss"]
    legal = state["adv_memory"][0]["legal"]
    assert legal.shape[1] == 3 and legal[:, 1].all() and not legal[:, 0].all()  # call always legal, fold only facing a bet
    train_main(["--resume", str(tmp_path / "checkpoint.pt"), "--iterations", "3"] + common)
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["iteration"] == 3 and len(state["adv_memory"][1]["legal"]) == len(state["adv_memory"][1]["t"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reservoir_replacement_matches_sequential_algorithm_r(device):
    """Duplicate slots within one add: the result equals item-by-item Algorithm R with the same draws (the last item
    drawn for a slot stays), and every stored row keeps its own fields together (CUDA's indexed writes with duplicate
    indices pick an arbitrary winner per tensor)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    cap, n = 50, 2000
    buf = ReservoirBuffer(cap, device, obs_dim=31, legal_dim=4, seed=3)
    ref_rng = np.random.default_rng(3)
    ids = np.arange(n, dtype=np.float32)
    obs = np.zeros((n, 31), np.float32)
    obs[:, 30] = ids
    buf.add(obs, ids, np.repeat(ids[:, None], 4, 1), _bits(np.arange(n) % 16))
    ref = list(range(cap))
    m = np.arange(cap, n) + 1  # what add() draws: one uniform per item beyond the empty slots
    r = ref_rng.random(n - cap) * m
    for item, ri in zip(range(cap, n), r):
        if ri < cap:
            ref[int(ri)] = item
    t = buf.t.cpu().numpy()
    np.testing.assert_array_equal(t, np.array(ref, np.float32))
    np.testing.assert_array_equal(buf.obs[:, 30].cpu().numpy(), t)
    np.testing.assert_array_equal(buf.target.cpu().numpy(), np.repeat(t[:, None], 4, 1))
    np.testing.assert_array_equal(buf.legal.cpu().numpy(), _bits(t.astype(np.int64) % 16))


def test_masked_loss_can_be_switched_off_on_resume():
    from headsup.deepcfr.train import build_parser

    p = build_parser()
    assert p.parse_args(["--masked-loss"]).masked_loss and not p.parse_args(["--no-masked-loss"]).masked_loss
    dest = {a.dest for a in p._actions if "--no-masked-loss" in a.option_strings}
    assert dest == {"masked_loss"}  # main() detects it as given, so the checkpoint's value is not inherited


def test_buffer_checkpoints_hold_only_the_stored_rows(tmp_path):
    """Host-resident buffers: a slice of the storage is a view, and saving a view writes the whole capacity
    (267 MB for 1000 rows of a 1M buffer; ~32 GB per checkpoint with the paper preset)."""
    import os

    from headsup.deepcfr.memory import CircularBuffer

    buf = ReservoirBuffer(200_000, "cpu", obs_dim=31, legal_dim=4, seed=0)
    buf.add(np.zeros((10, 31), np.float32), np.ones(10, np.float32), np.zeros((10, 4), np.float32), np.ones((10, 4), bool))
    buf.save(tmp_path / "reservoir.pt")
    assert os.path.getsize(tmp_path / "reservoir.pt") < 100_000
    fifo = CircularBuffer(200_000, "cpu", obs_dim=86)
    fifo.add(np.zeros((10, 86), np.float32), np.zeros(10), np.zeros(10, np.float32))
    torch.save(fifo.state_dict(), tmp_path / "fifo.pt")
    assert os.path.getsize(tmp_path / "fifo.pt") < 100_000
    other = ReservoirBuffer(200_000, "cpu", obs_dim=31, legal_dim=4).load(tmp_path / "reservoir.pt")
    assert len(other) == 10 and other.legal[:10].all()


def test_circular_buffer_loads_a_wrapped_fifo_in_order():
    """A wrapped FIFO loaded into a larger (or smaller) buffer keeps exactly its rows, oldest first: with the saved
    head kept, new rows overwrote old ones while the size grew over rows that were never written."""
    from headsup.deepcfr.memory import CircularBuffer

    def rows(n0, n1):
        t = np.arange(n0, n1, dtype=np.float32)
        return np.repeat(t[:, None], 3, 1), np.arange(n0, n1) % 4, t

    small = CircularBuffer(10, "cpu", obs_dim=3)
    small.add(*rows(0, 7))
    small.add(*rows(7, 25))  # holds 15..24, wrapped: the oldest row sits at position 7
    assert small.head == 7 and small.target[7] == 15
    state = small.state_dict()
    big = CircularBuffer(16, "cpu", obs_dim=3)
    big.load_state_dict(state)
    assert big.size == 10 and big.target[:10].tolist() == list(range(15, 25))
    big.add(*rows(25, 28))
    assert big.size == 13 and sorted(big.target[:13].tolist()) == list(range(15, 28))
    assert (big.obs[:13, 0] == big.target[:13]).all() and (big.action[:13] == big.target[:13].long() % 4).all()
    big.add(*rows(28, 40))  # wraps: the oldest rows go first
    assert sorted(big.target[:16].tolist()) == list(range(24, 40))
    tiny = CircularBuffer(4, "cpu", obs_dim=3)
    tiny.load_state_dict(state)  # a smaller buffer keeps the newest rows
    assert tiny.size == 4 and sorted(tiny.target[:4].tolist()) == [21, 22, 23, 24]
    same = CircularBuffer(10, "cpu", obs_dim=3)
    same.load_state_dict(state)
    same.add(*rows(25, 27))
    assert sorted(same.target[:10].tolist()) == list(range(17, 27))


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_interrupt_inside_an_iteration_keeps_the_checkpoint_consistent(tmp_path, monkeypatch):
    """Ctrl-C between the two seats' fits: the checkpoint and the SD-CFR bank must describe the last COMPLETE
    iteration (it saved an advanced counter and one more iterate for seat 0; every later evaluation / resume crashed)."""
    import headsup.deepcfr.train as train

    real, calls = train.train_advantage_net, []

    def interrupting(*args, **kwargs):
        calls.append(1)
        if len(calls) == 6:  # iteration 3, seat 1 (two fits per iteration)
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "train_advantage_net", interrupting)
    common = ["--algo", "both", "--workers", "2", "--device", "cpu", "--no-compile", "--traversals", "30", "--value-steps", "3",
              "--batch-size", "64", "--policy-epochs", "1", "--eval-hands", "0", "--eval-every", "0", "--policy-eval-every", "0",
              "--lbr-every", "0", "--lbr-final-hands", "0", "--no-tensorboard", "--adv-capacity", "20000", "--strat-capacity", "20000",
              "--checkpoint-every", "1", "--out", str(tmp_path)]
    train.main(["--iterations", "5"] + common)
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["iteration"] == 2
    assert [next(iter(d.values())).shape[0] for _, d in sorted(state["iterates"].items())] == [3, 3]  # untrained + 2 iterations each
    bank = torch.load(tmp_path / "iterates.pt", map_location="cpu", weights_only=True)
    assert bank["T"] == 3 and bank["iterations"] == [0, 1, 2]  # the untrained nets, then one per iteration
    monkeypatch.setattr(train, "train_advantage_net", real)
    train.main(["--resume", str(tmp_path / "checkpoint.pt"), "--iterations", "4"] + common)
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["iteration"] == 4 and torch.load(tmp_path / "iterates.pt", map_location="cpu", weights_only=True)["T"] == 5


_TINY = ["--workers", "2", "--device", "cpu", "--no-compile", "--traversals", "30", "--value-steps", "3", "--batch-size", "64",
         "--policy-epochs", "1", "--eval-hands", "0", "--eval-every", "0", "--policy-eval-every", "0", "--lbr-every", "0",
         "--lbr-final-hands", "0", "--no-tensorboard", "--adv-capacity", "20000", "--strat-capacity", "20000", "--q-steps", "3",
         "--q-batch", "32"]


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_resume_takes_explicit_flags_in_every_spelling_and_continues_in_place(tmp_path):
    import headsup.deepcfr.train as train

    out = tmp_path / "run"
    train.main(["--algo", "sdcfr", "--iterations", "1", "--checkpoint-every", "1", "--policy-steps", "7", "--out", str(out)] + _TINY)
    ck = str(out / "checkpoint.pt")
    a = train.cli_args(["--resume", ck])
    assert (a.traversals, a.value_steps, a.checkpoint_every, a.policy_steps) == (30, 3, 1, 7)  # inherited, incl. the checkpoint cadence
    assert a.out == str(out)  # a resumed run continues in the checkpoint's directory unless --out is given
    assert train.cli_args(["--resume", ck, "--out", "elsewhere"]).out == "elsewhere"
    for argv in (["--traversals", "9", "--value-steps", "5"], ["--traversals=9", "--value-steps=5"], ["--trav", "9", "--value-s", "5"]):
        a = train.cli_args(["--resume", ck] + argv)
        assert (a.traversals, a.value_steps) == (9, 5), argv
    a = train.cli_args(["--resume", ck, "--policy-epochs", "5"])  # explicit epochs beat the checkpoint's step count
    assert (a.policy_epochs, a.policy_steps) == (5, None)
    assert train.cli_args(["--resume", ck, "--policy-steps", "11"]).policy_steps == 11


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_resume_with_another_algorithm_and_new_learning_rate(tmp_path, capsys):
    """A checkpoint resumed with another --algo used to crash on the value-net scales (IndexError); the DREAM value
    optimisers kept the checkpoint's learning rate; the traversal seeds restarted from the first iteration's."""
    import headsup.deepcfr.train as train

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    train.main(["--algo", "sdcfr", "--iterations", "2", "--checkpoint-every", "1", "--out", str(a_dir)] + _TINY)
    train.main(["--resume", str(a_dir / "checkpoint.pt"), "--algo", "dream", "--iterations", "3", "--out", str(a_dir)] + _TINY)
    state = torch.load(a_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["iteration"] == 3 and len(state["value_nets"]) == 2
    train.main(["--algo", "both", "--iterations", "1", "--checkpoint-every", "1", "--out", str(b_dir)] + _TINY)
    train.main(["--resume", str(b_dir / "checkpoint.pt"), "--algo", "escher", "--iterations", "2", "--out", str(b_dir)] + _TINY)
    capsys.readouterr()
    # sdcfr -> both: the new strategy memory starts empty - say so
    train.main(["--algo", "sdcfr", "--iterations", "1", "--checkpoint-every", "1", "--out", str(tmp_path / "c")] + _TINY)
    train.main(["--resume", str(tmp_path / "c" / "checkpoint.pt"), "--algo", "both", "--iterations", "2", "--out", str(tmp_path / "c")] + _TINY)
    assert "no strategy memory" in capsys.readouterr().out
    # dream: --lr given on resume reaches the persistent value-net optimisers; the seed stream does not restart
    trainer = train.DeepCFRTrainer(train.cli_args(["--resume", str(a_dir / "checkpoint.pt"), "--lr", "0.0003", "--out", str(a_dir)] + _TINY))
    first = [trainer._seed() for _ in range(4)]
    trainer.load_checkpoint(str(a_dir / "checkpoint.pt"))
    assert [g["lr"] for o in trainer.value_opts for g in o.param_groups] == [0.0003, 0.0003]
    assert not set(first) & {trainer._seed() for _ in range(4)}
    trainer.runner.close()


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_resume_continues_in_the_checkpoints_game_not_in_the_presets(tmp_path, monkeypatch):
    """A resumed run rebuilt its game from the preset NAME: after a preset changes (FHP's flop cap), an older
    checkpoint must keep training in the game its networks and memories belong to."""
    import headsup.deepcfr.train as train
    import headsup.games.holdem as holdem
    from headsup.game import FHP

    old_fhp = FHP.with_(raise_caps=(3, 3))
    monkeypatch.setitem(holdem.PRESETS, "fhp", old_fhp)
    out = tmp_path / "run"
    train.main(["--game", "fhp", "--algo", "sdcfr", "--iterations", "1", "--checkpoint-every", "1", "--out", str(out)] + _TINY)
    monkeypatch.setitem(holdem.PRESETS, "fhp", FHP)  # the preset moves on
    trainer = train.DeepCFRTrainer(train.cli_args(["--resume", str(out / "checkpoint.pt"), "--iterations", "2"] + _TINY))
    assert trainer.game.raise_caps == (3, 3) and trainer.game == old_fhp
    trainer.load_checkpoint(str(out / "checkpoint.pt"))
    trainer.runner.close()


def test_paper_preset_selects_the_papers_network():
    import headsup.deepcfr.train as train

    a = train.cli_args(["--preset", "paper", "--game", "fhp"])
    assert (a.net, a.rm_fallback, a.loss_weights) == ("deepcfr", "argmax", "paper")
    assert train.cli_args([]).net == "current"
    assert train.cli_args(["--preset", "paper", "--net", "paper"]).net == "paper"  # an explicit flag wins


def test_loss_weights_are_rescaled_by_two_over_T():
    """Paper 5.3: "we rescale all the batch weights by 2/T".  With the raw weights t the loss and its gradient grow
    with the iteration, and the gradient clip at 1 rescales every single step (measured: norms 25-460)."""
    from headsup.deepcfr.train import loss_weight_scale, train_advantage_net

    assert loss_weight_scale(450, 1.0) == pytest.approx(2 / 450) and loss_weight_scale(450, 1.0, "raw") == 1.0
    assert loss_weight_scale(10, 2.0) == pytest.approx(3 / 100)  # t^p: the weights average ~1 over iterations 1..T
    assert loss_weight_scale(0, 1.0) == 1.0  # before the first iteration
    rng = np.random.default_rng(0)
    obs = rng.random((200, 31), dtype=np.float32)
    obs[:, :23] = rng.integers(0, 3, (200, 23))
    target = np.tile(np.array([1.0, -1.0, 1.0, -1.0], np.float32), (200, 1))
    first = {}
    for mode in ("paper", "raw"):
        buf = ReservoirBuffer(1000, "cpu", obs_dim=31, seed=0)
        buf.add(obs, np.full(200, 40.0, np.float32), target)
        losses = []
        torch.manual_seed(0)
        train_advantage_net(buf, "cpu", steps=2, batch_size=64, compile=False, iteration=40, loss_weights=mode, zero_head=True,
                            log=lambda tag, value, step: losses.append((tag, value)))
        first[mode] = next(v for tag, v in losses if tag.endswith("/loss"))
    assert first["paper"] == pytest.approx(2.0, rel=1e-4) and first["raw"] == pytest.approx(40.0, rel=1e-4)  # (2 / T) t = 2 at t = T


def test_refits_start_from_a_random_head():
    """Paper 5.2: every network is trained "from scratch ... starting from a random initialization"; only the first
    strategy needs the all-zero output (ours started every refit with a zero head)."""
    from headsup.deepcfr.train import train_advantage_net

    assert not BaseModel().action_head.weight.any() and BaseModel(zero_head=False).action_head.weight.abs().mean() > 0.01
    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, _, _ = run_traversals_python(w, 0, 50, 1.0, seed=0)
    buf = ReservoirBuffer(10000, "cpu", obs_dim=31, seed=0)
    buf.add(adv.obs, adv.t, adv.target)
    net, _ = train_advantage_net(buf, "cpu", steps=1, batch_size=64, compile=False, target_scale=None)
    assert net.action_head.weight.abs().mean() > 0.01  # one step from a zero head moves each weight by ~lr = 0.001
    empty, _ = train_advantage_net(ReservoirBuffer(100, "cpu", obs_dim=31, seed=0), "cpu", steps=1, batch_size=64, compile=False)
    assert not empty.action_head.weight.any()  # no samples yet: the uniform net


def test_policy_fit_follows_the_authors(monkeypatch):
    """Constant learning rate (StepLR x0.9 every 2 % of the fit left 20 % of the lr integral), gradient clipping, and
    with legal masks a softmax over the legal actions only (as the strategy is used)."""
    import headsup.deepcfr.train as train

    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    _, strat, _ = run_traversals_python(w, 0, 100, 1.0, seed=0)
    buf = ReservoirBuffer(10000, "cpu", obs_dim=31, seed=0)
    buf.add(strat.obs, strat.t, strat.target)
    seen = []
    step = train._run_step

    def spy(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal=None):
        seen.append((opt.param_groups[0]["lr"], grad_clip))
        return step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal)

    monkeypatch.setattr(train, "_run_step", spy)
    train.train_policy_net(buf, "cpu", epochs=None, steps=120, batch_size=32, compile=False, progress=False)
    assert len(seen) == 120 and set(seen) == {(1e-3, 1.0)}
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    target = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    legal = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    t = torch.ones(2)
    assert float(train._policy_loss(masked=True)(logits, t, target, legal)) == pytest.approx(0.0, abs=1e-12)
    assert float(train._policy_loss()(logits, t, target)) > 0.1


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_masked_loss_also_masks_the_policy_fit(tmp_path):
    """--masked-loss: the strategy memory keeps the legal masks of its samples and the policy net is fitted with them."""
    from headsup.deepcfr.train import main as train_main

    train_main(["--game", "fhp", "--algo", "both", "--masked-loss", "--net", "deepcfr", "--iterations", "2", "--checkpoint-every", "1",
                "--out", str(tmp_path)] + _TINY)
    state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    legal, target = state["strat_memory"]["legal"], state["strat_memory"]["target"]
    assert legal.shape == target.shape and legal.shape[1] == 3 and legal[:, 1].all() and not legal[:, 0].all()
    assert not target[legal == 0].any()  # a strategy never plays a masked action
    assert (tmp_path / "policy.pth").exists()


def test_python_samplers_never_pick_a_zero_probability_action():
    """The reference samplers mirror the C++ rule: when the partial sums stop short of the draw, the remainder
    belongs to the last action WITH probability (min(index, n - 1) picked the illegal last action)."""
    from headsup.deepcfr.traverse import sample_action

    p = np.array([0.5, 0.49999994, 0.0], dtype=np.float32)
    assert sample_action(p, 0.99999999) == 1 and sample_action(p, 0.2) == 0 and sample_action(p, 0.7) == 1
    assert sample_action(np.array([0.0, 1.0, 0.0]), 0.0) == 1
    if native.available():
        rng = np.random.default_rng(0)
        for _ in range(300):
            q = rng.random(4).astype(np.float32) * (rng.random(4) < 0.7)
            if q.sum() == 0:
                continue
            q /= q.sum()
            u = float(np.float32(rng.random()))
            a = sample_action(q, u)
            assert q[a] > 0 and abs(a - native.module().sample_index(q, u)) <= (abs(np.cumsum(q) - u).min() < 1e-6)


def test_argmax_fallback_plays_uniformly_over_exactly_tied_actions():
    """The "argmax" fallback of the DeepCFR paper leaves ties undefined; taking the first index made the untrained
    (all-zero) network fold to every bet in iteration 1.  The authors play uniformly before the first network
    exists: exact ties share the probability, anything else is the single best action as before."""
    from headsup.algos.deep import regret_matching_np, regret_matching_rows
    from headsup.players import regret_matching_torch

    cases = [  # advantages, legal, expected
        ([0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25]),
        ([0.0, 0.0, 0.0, 0.0], [0, 1, 1, 1], [0, 1 / 3, 1 / 3, 1 / 3]),
        ([-1.0, -1.0, -3.0, -2.0], [1, 1, 1, 1], [0.5, 0.5, 0, 0]),
        ([-1.0, -0.5, -3.0, -0.5], [1, 0, 1, 1], [0, 0, 0, 1]),  # the tied best action that is legal
        ([-1.0, -3.0, -2.0, -0.5], [1, 1, 1, 1], [0, 0, 0, 1]),
        ([1.0, 3.0, -2.0, 0.0], [1, 1, 1, 1], [0.25, 0.75, 0, 0]),
    ]
    for adv, legal, want in cases:
        adv, legal = np.array(adv, np.float32), np.array(legal, bool)
        np.testing.assert_allclose(regret_matching(adv, legal=legal, fallback="argmax"), want, atol=1e-7)
        np.testing.assert_allclose(regret_matching_torch(torch.tensor(adv)[None], legal[None], "argmax")[0].numpy(), want, atol=1e-7)
        np.testing.assert_allclose(regret_matching_np(adv.astype(np.float64), legal, True), want, atol=1e-7)
        np.testing.assert_allclose(regret_matching_rows(adv.astype(np.float64)[None], legal[None], True)[0], want, atol=1e-7)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_untrained_networks_play_uniformly_in_the_first_iteration():
    """C++ and Python samplers: with untrained argmax-fallback networks the opponent's strategy rows are uniform over
    the legal actions (they were "fold facing a bet, else check": seat 1 got no samples against the untrained seat 0)."""
    from headsup.game import FHP

    cpp = native.module()
    w = BaseModel(arch="deepcfr", game=FHP, rm_fallback="argmax").numpy_weights()
    net = native.make_model(w)
    decks = np.stack([np.random.default_rng(i).permutation(52)[:9] for i in range(40)]).astype(np.int32)
    out = cpp.run_traversals(net, net, 1, 40, 1.0, 0, native.engine_config(game=FHP), decks)
    sigma, legal = np.asarray(out[5]), np.asarray(out[8])
    assert len(sigma) > 0 and len(out[0]) > 0  # the traverser reaches its own decisions
    np.testing.assert_allclose(sigma, legal / legal.sum(1, keepdims=True), atol=1e-7)
    _, strat, _ = run_traversals_python([w, w], 1, 40, 1.0, seed=0, decks=decks)
    np.testing.assert_allclose(strat.target, strat.legal / strat.legal.sum(1, keepdims=True), atol=1e-7)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_bank_started_from_a_checkpoint_without_one_knows_its_iterations(tmp_path):
    """Resuming a Deep CFR checkpoint as SD-CFR: the bank starts with the checkpoint's networks as the networks of
    its iteration (it started with fresh untrained ones, so iteration 3's network was weighted like iteration 1's)."""
    import headsup.deepcfr.train as train

    out = tmp_path / "run"
    train.main(["--algo", "deepcfr", "--iterations", "2", "--checkpoint-every", "1", "--out", str(out)] + _TINY)
    train.main(["--resume", str(out / "checkpoint.pt"), "--algo", "sdcfr", "--iterations", "4"] + _TINY)
    bank = torch.load(out / "iterates.pt", map_location="cpu", weights_only=True)
    assert bank["iterations"] == [2, 3, 4]
    state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["iterate_first"] == 2
    head = "action_head.weight"
    assert bank["seats"][0][head][0].abs().sum() > 0  # the first stored net is the trained one of iteration 2


def test_dream_and_escher_defaults_follow_their_papers():
    """DREAM 5: "picking the action with the highest advantage with probability 1 when all are negative"; both
    authors' networks mask illegal outputs (ours fitted them to zero targets by default)."""
    import headsup.deepcfr.train as train

    for algo in ("dream", "escher"):
        a = train.cli_args(["--algo", algo])
        assert (a.rm_fallback, a.masked_loss) == ("argmax", True)
        a = train.cli_args(["--algo", algo, "--rm-fallback", "uniform", "--no-masked-loss"])
        assert (a.rm_fallback, a.masked_loss) == ("uniform", False)
    a = train.cli_args(["--algo", "sdcfr"])
    assert (a.rm_fallback, a.masked_loss) == ("uniform", False)
    # ESCHER's Table 3 as a preset; explicit flags win
    a = train.cli_args(["--algo", "escher", "--preset", "escher", "--q-batch", "256"])
    assert (a.traversals, a.value_trajectories, a.batch_size, a.value_steps, a.q_steps, a.q_batch, a.policy_steps, a.policy_batch_size) == (
        1000, 1000, 2048, 5000, 5000, 256, 10000, 2048)


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_escher_value_net_sees_all_trajectories_of_its_iteration(tmp_path):
    """ESCHER's value net is refitted on this iteration's trajectories; they went through the DREAM baseline's FIFO
    (--q-capacity), which silently kept the newest rows only."""
    import headsup.deepcfr.train as train

    trainer = train.DeepCFRTrainer(train.cli_args(["--algo", "escher", "--q-capacity", "40", "--iterations", "1", "--out", str(tmp_path)] + _TINY))
    seen = []
    collect = trainer.runner.collect_escher_values

    def spy(*args, **kw):
        val, nodes = collect(*args, **kw)
        seen.append(len(val))
        return val, nodes

    trainer.runner.collect_escher_values = spy
    trainer.cfr_iteration()
    assert seen[0] > 40 and len(trainer.value_memory[0]) == seen[0]
    trainer.runner.close()


def test_advantage_fit_can_anneal_the_learning_rate_and_average_its_weights(monkeypatch):
    """Two refits of one regret net on the same memory disagree (FHP at t = 125: 6.7 chips rms per net at the pre-flop
    infosets, where a tabular mean of the same samples has a standard error of 4.5), and regret matching turns that
    noise into the strategy.  A cosine learning rate, or the average of the weights over the second half of the fit,
    brings it to 2.7-2.9 at the same number of steps.  Not in the paper (constant rate, last weights): off by default."""
    import headsup.deepcfr.train as train

    w = [BaseModel().numpy_weights(), BaseModel().numpy_weights()]
    adv, _, _ = run_traversals_python(w, 0, 100, 1.0, seed=0)
    buf = ReservoirBuffer(10000, "cpu", obs_dim=31, seed=0)
    buf.add(adv.obs, adv.t, adv.target)
    rates, weights = [], []
    step = train._run_step

    def spy(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal=None):
        rates.append(opt.param_groups[0]["lr"])
        out = step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal)
        weights.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return out

    monkeypatch.setattr(train, "_run_step", spy)
    kw = dict(steps=20, batch_size=64, compile=False, target_scale=None)
    train.train_advantage_net(buf, "cpu", **kw)
    assert set(rates) == {1e-3}  # the paper: a constant rate
    rates.clear()
    train.train_advantage_net(buf, "cpu", lr_schedule="cosine", **kw)
    assert rates == pytest.approx([1e-3 * 0.5 * (1 + np.cos(np.pi * i / 20)) for i in range(20)]) and rates[-1] < 1e-5
    with pytest.raises(ValueError, match="schedule"):
        train.train_advantage_net(buf, "cpu", lr_schedule="linear", **kw)

    weights.clear()
    last, _ = train.train_advantage_net(buf, "cpu", **kw)
    assert all(torch.equal(v, weights[-1][k]) for k, v in last.state_dict().items())  # default: the last weights
    weights.clear()
    net, _ = train.train_advantage_net(buf, "cpu", weight_average=0.9, **kw)
    want = {k: v.clone() for k, v in weights[9].items()}  # the average starts at the middle of the fit ...
    for snap in weights[10:]:
        for k in want:
            want[k] = 0.9 * want[k] + 0.1 * snap[k]  # ... as an exponential moving average
    assert not net.training
    for k, v in net.state_dict().items():
        assert torch.allclose(v, want[k], atol=1e-7), k
    assert any(not torch.allclose(v, weights[-1][k], atol=1e-5) for k, v in net.state_dict().items())


def test_fit_options_reach_the_fit_and_are_inherited_on_resume(tmp_path, monkeypatch):
    import headsup.deepcfr.train as train

    assert (train.cli_args(_TINY).lr_schedule, train.cli_args(_TINY).weight_average) == ("constant", 0.0)
    seen = []
    fit = train.train_advantage_net

    def spy(*args, **kw):
        seen.append((kw.get("lr_schedule"), kw.get("weight_average")))
        return fit(*args, **kw)

    monkeypatch.setattr(train, "train_advantage_net", spy)
    out = tmp_path / "run"
    train.main(["--algo", "sdcfr", "--iterations", "1", "--checkpoint-every", "1", "--lr-schedule", "cosine", "--weight-average", "0.5",
                "--out", str(out)] + _TINY)
    assert seen and set(seen) == {("cosine", 0.5)}
    a = train.cli_args(["--resume", str(out / "checkpoint.pt")])
    assert (a.lr_schedule, a.weight_average) == ("cosine", 0.5)


def test_policy_fit_has_its_own_learning_rate(tmp_path, monkeypatch):
    """--lr sets both fits; a larger rate with a cosine schedule suits the advantage fit, and --policy-lr keeps the
    average-strategy fit (constant rate) at its own."""
    import headsup.deepcfr.train as train

    assert train.cli_args(_TINY).policy_lr is None
    seen = {}
    fits = {"adv": train.train_advantage_net, "policy": train.train_policy_net}
    monkeypatch.setattr(train, "train_advantage_net", lambda *a, **kw: (seen.setdefault("adv", a[4]), fits["adv"](*a, **kw))[1])
    monkeypatch.setattr(train, "train_policy_net", lambda *a, **kw: (seen.setdefault("policy", a[4]), fits["policy"](*a, **kw))[1])
    out = tmp_path / "run"
    train.main(["--algo", "both", "--iterations", "1", "--checkpoint-every", "1", "--lr", "0.003", "--policy-lr", "0.0005",
                "--out", str(out)] + _TINY)
    assert seen == {"adv": 0.003, "policy": 0.0005}
    assert train.cli_args(["--resume", str(out / "checkpoint.pt")]).policy_lr == 0.0005
    seen.clear()
    train.main(["--algo", "both", "--iterations", "1", "--lr", "0.003", "--out", str(tmp_path / "run2")] + _TINY)
    assert seen == {"adv": 0.003, "policy": 0.003}  # default: the same rate


def test_strategy_memory_can_live_on_another_device_than_the_advantage_memories(tmp_path):
    """The paper's 40 M-sample memories: two advantage memories fill a 24 GB card; the strategy memory, read only for the
    policy fits, can stay in host RAM (--strat-memory-device)."""
    import headsup.deepcfr.train as train

    assert train.cli_args(_TINY).strat_memory_device is None
    args = train.cli_args(["--algo", "both", "--iterations", "1", "--strat-memory-device", "cpu", "--out", str(tmp_path / "run")] + _TINY)
    trainer = train.DeepCFRTrainer(args)
    assert trainer.strat_memory.device == torch.device("cpu") and trainer.strat_memory.sample_device == trainer.device
    assert all(m.device == trainer.device for m in trainer.adv_memory)
    args = train.cli_args(["--algo", "both", "--iterations", "1", "--memory-device", "cpu", "--out", str(tmp_path / "run2")] + _TINY)
    trainer = train.DeepCFRTrainer(args)  # without the option the strategy memory follows --memory-device
    assert trainer.strat_memory.device == torch.device("cpu") and all(m.device == torch.device("cpu") for m in trainer.adv_memory)


def test_restoring_a_memory_moves_the_saved_rows_in_pieces(monkeypatch):
    """A checkpoint's rows go to the buffer's device piece by piece: moving a whole saved tensor needs its rows a second
    time on the device, which a nearly full GPU does not have (resuming 20 M rows into 40 M-row memories on a 24 GB card
    failed with "CUDA driver error: device not ready")."""
    from headsup.deepcfr import memory

    rng = np.random.default_rng(0)
    obs = rng.random((50, memory.OBS_DIM)).astype(np.float32)
    obs[:, : memory.OBS_INT_DIM] = 1.0
    small = ReservoirBuffer(64, "cpu", seed=0, legal_dim=memory.NUM_ACTIONS)
    small.add(obs, np.arange(50, dtype=np.float32), rng.random((50, memory.NUM_ACTIONS)).astype(np.float32), rng.random((50, memory.NUM_ACTIONS)) < 0.5)
    state = small.state_dict()

    moved = []
    to = torch.Tensor.to
    monkeypatch.setattr(memory, "LOAD_CHUNK", 16, raising=False)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: (moved.append(len(self)), to(self, *a, **k))[1])
    big = ReservoirBuffer(128, "cpu", seed=0, legal_dim=memory.NUM_ACTIONS)
    big.load_state_dict(state)
    monkeypatch.undo()

    assert moved and max(moved) <= 16
    assert big.size == 50 and big.seen == small.seen
    assert torch.equal(big.obs, small.obs) and torch.equal(big.t[:50], small.t[:50])
    assert torch.equal(big.target[:50], small.target[:50]) and torch.equal(big.legal[:50], small.legal[:50])
