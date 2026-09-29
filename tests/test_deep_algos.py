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
