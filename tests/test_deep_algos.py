"""Deep CFR / SD-CFR / DREAM / ESCHER on the game protocol (Kuhn), against exact exploitability."""

import pytest

from headsup.algos.best_response import exploitability
from headsup.algos.deep import DeepSolver
from headsup.games import UniformPolicy, make_game


@pytest.mark.parametrize("algo", ["deepcfr", "sdcfr", "dream", "escher"])
def test_deep_algorithms_reduce_kuhn_exploitability(algo):
    g = make_game("kuhn")
    uniform = exploitability(g, UniformPolicy(g))[0]
    # ESCHER re-fits its value net from scratch every iteration (as the reference code): more data / steps
    its, trav, q_steps = (20, 200, 300) if algo == "escher" else (12, 60, 60)
    s = DeepSolver(g, algo, traversals=trav, adv_steps=120, adv_batch=256, policy_steps=200, policy_batch=256,
                   q_steps=q_steps, q_batch=128, seed=0)
    s.iterate(its)
    ev = s.evaluate()
    assert ev["average"] < 0.6 * uniform, ev  # uniform: 0.458; the average strategies reach ~0.05-0.25 here
    assert len(s.iterates[0]) == its + 1 and s.stats["iteration"] == its
    # the exact SD-CFR average of the bank equals the reach-weighted mixture (sums to one everywhere)
    if algo in ("sdcfr", "dream"):
        pol = s.average_policy()
        for key, probs in pol.table.items():
            assert probs.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_graph_captured_fits_do_not_grow_gpu_memory():
    """Each fit is a captured CUDA graph; repeated fits must not accumulate GPU memory (a new warm-up
    stream per fit leaked a cuBLAS workspace, ~20 MB, every time)."""
    import torch

    from headsup.algos.deep import _fit_from_buffer

    s = DeepSolver(make_game("leduc"), "sdcfr", traversals=50, adv_steps=20, adv_batch=256, device="cuda", seed=0)
    s.iterate(1)
    _fit_from_buffer(s._new_model(), s.adv_memory[0], 20, 256, 1e-3, s.device)
    before = torch.cuda.memory_reserved()
    for _ in range(10):
        _fit_from_buffer(s._new_model(), s.adv_memory[0], 20, 256, 1e-3, s.device)
    assert torch.cuda.memory_reserved() - before < 8 << 20


def test_checkpoint_resume_continues_the_run(tmp_path):
    from headsup.algos.deep import main

    ck = tmp_path / "ck.pt"
    args = ["--game", "kuhn", "--algo", "dream", "--traversals", "20", "--adv-steps", "10", "--adv-batch", "64", "--q-steps", "5",
            "--eval-every", "2", "--checkpoint", str(ck), "--json", str(tmp_path / "c.json")]
    main(args + ["--iterations", "2"])
    main(args + ["--iterations", "4"])  # resumes at iteration 2
    import json

    curve = json.load(open(tmp_path / "c.json"))["curve"]
    assert [c["iteration"] for c in curve] == [2, 4]


def test_deepcfr_architecture_for_the_small_games():
    import torch

    g = make_game("leduc")
    for in_dim in (g.obs_dim, 2 * g.obs_dim):  # one infostate / both players' (history nets)
        m = g.make_model(arch="deepcfr", in_dim=in_dim)
        out = m(torch.randn(7, in_dim))
        assert out.shape == (7, g.num_actions) and torch.all(out == 0)  # zero-initialised head: uniform start
    s = DeepSolver(make_game("kuhn"), "sdcfr", traversals=20, adv_steps=10, adv_batch=64, seed=0, model_kwargs={"arch": "deepcfr"})
    s.iterate(2)
    assert 0 <= s.evaluate()["average"] < 1


@pytest.mark.parametrize("name", ["kuhn", "leduc"])
def test_legal_mask_from_info_state_matches_the_game(name):
    import numpy as np
    import torch

    from headsup.games.leduc import legal_mask_from_info_state

    g = make_game(name)
    rows, masks = [], []

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        rows.append(s.info_state(s.current_player))
        masks.append(s.legal_mask())
        for a in s.legal_actions():
            walk(s.child(a))

    walk(g.new_initial_state())
    got = legal_mask_from_info_state(g, torch.as_tensor(np.stack(rows))).numpy() > 0
    np.testing.assert_array_equal(got, np.stack(masks))


def test_pokerrl_nets_mask_illegal_actions():
    import torch

    g = make_game("leduc")
    s = DeepSolver(g, "deepcfr", traversals=20, adv_steps=5, adv_batch=64, policy_steps=5, policy_batch=64, seed=0,
                   model_kwargs={"arch": "pokerrl"}, normalized_weights=True, grad_clip=10.0, mean_regret=True)
    s.iterate(2)
    x = torch.as_tensor(s._info_obs)
    legal = torch.as_tensor(s._info_legal)
    with torch.no_grad():
        assert torch.all(s.nets[0](x)[~legal] == 0)  # dueling advantages: exactly 0 where illegal
        assert torch.all(s._new_model(policy=True)(x)[~legal] < -1e19)
    assert 0 <= s.evaluate()["average"] < 3


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_pokerrl_nets_train_inside_cuda_graphs():
    """The legal-action mask is recomputed inside the network: no host-to-device copies during capture."""
    s = DeepSolver(make_game("leduc"), "deepcfr", traversals=20, adv_steps=8, adv_batch=64, policy_steps=8, policy_batch=64,
                   device="cuda", seed=0, model_kwargs={"arch": "pokerrl"})
    s.iterate(2)
    assert 0 <= s.evaluate()["average"] < 3
