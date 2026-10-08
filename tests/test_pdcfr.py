"""Deep (Predictive) Discounted CFR (Xu et al. 2025) on Kuhn / Leduc: pieces and end-to-end convergence."""

import numpy as np
import pytest
import torch

from headsup.algos.best_response import expected_value, exploitability
from headsup.algos.pdcfr import CHANCE, DECISION, TERMINAL, Tree, discount, mlp, strategy_rows
from headsup.games import UniformPolicy, make_game


def test_discount_schedule():
    assert discount(1, 2.0, 1.5) == 0.0 and discount(1, 2.3, 1.0) == 0.0
    assert discount(2, 2.0, 1.5) == pytest.approx(1 / 2.5)
    assert discount(3, 2.3, 1.0) == pytest.approx(2**2.3 / (2**2.3 + 1))
    assert discount(500, 2.0, 1.5) == pytest.approx(1.0, abs=1e-5)


def test_strategy_rows_variants_and_fallbacks():
    legal = np.array([[True, True, True], [False, True, True], [False, True, True], [False, True, True]])
    R = np.array([[1.0, 3.0, -2.0], [9.0, -1.0, -3.0], [0.0, -1.0, -3.0], [5.0, 2.0, -4.0]])
    r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -2.0, -1.0], [0.0, -3.0, 1.0]])
    s = strategy_rows(R, r, legal, 0.5, "dcfr+", "authors")
    np.testing.assert_allclose(s[0], [0.25, 0.75, 0.0])  # regret matching on the positive part
    np.testing.assert_allclose(s[1], [0.0, 1.0, 0.0])  # nothing positive among the legal: the largest raw output
    np.testing.assert_allclose(s[3], [0.0, 1.0, 0.0])  # the illegal action's output is ignored
    p = strategy_rows(R, r, legal, 0.5, "pdcfr+", "authors")
    np.testing.assert_allclose(p[0], [0.25, 0.75, 0.0])  # max(R, 0) d + r, clipped
    np.testing.assert_allclose(p[2], [0.0, 1.0, 0.0])  # all clipped to zero: the authors' code plays the first legal action
    np.testing.assert_allclose(p[3], [0.0, 0.0, 1.0])  # prediction: 2 * 0.5 - 3 < 0 and 0 + 1 > 0
    np.testing.assert_allclose(strategy_rows(R, r, legal, 0.5, "pdcfr+", "argmax")[2], [0.0, 0.0, 1.0])  # largest unclipped prediction
    np.testing.assert_allclose(strategy_rows(R, r, legal, 0.5, "pdcfr+", "uniform")[2], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(strategy_rows(R, r, legal, 0.5, "dcfr+", "uniform")[1], [0.0, 0.5, 0.5])


def test_mlp_starts_at_zero_with_truncated_normal_hidden_layers():
    torch.manual_seed(0)
    net = mlp(34, 3)
    assert [m.out_features for m in net if isinstance(m, torch.nn.Linear)] == [64, 64, 64, 3]
    with torch.no_grad():
        assert float(net(torch.randn(5, 34)).abs().max()) == 0.0
    w = net[0].weight.detach()
    assert float(w.abs().max()) <= 2 / 34**0.5 + 1e-6 and float(net[0].bias.detach().abs().max()) == 0.0


@pytest.mark.parametrize("name,nodes,infosets,terminals,depth,scale", [("kuhn", 58, 12, 30, 3, 2.0), ("leduc", 9457, 936, 5520, 8, 13.0)])
def test_tree_matches_the_game(name, nodes, infosets, terminals, depth, scale):
    g = make_game(name)
    t = Tree(g)
    assert (t.num_nodes, t.num_infosets, int((t.kind == TERMINAL).sum()), t.depth, t.max_utility) == (nodes, infosets, terminals, depth, scale)
    assert t.hist_x.shape == (t.num_decisions, 2 * g.obs_dim) and t.info_obs.shape == (infosets, g.obs_dim)
    dec = t.kind == DECISION
    assert (t.dec[dec] >= 0).all() and (t.dec[~dec] == -1).all() and (t.info[dec] >= 0).all()
    np.testing.assert_array_equal(t.legal[dec], t.info_legal[t.info[dec]])  # an infoset's histories share its legal actions
    np.testing.assert_array_equal(t.player[dec], t.info_player[t.info[dec]])
    assert (t.child[t.legal] > np.nonzero(t.legal)[0]).all()  # children come after their parents
    ch = t.kind == CHANCE
    np.testing.assert_allclose(t.chance_prob[ch].sum(1), 1.0)
    # exact values: the uniform profile's value of the game equals the reference implementation's
    sigma = t.info_legal / t.info_legal.sum(1, keepdims=True)
    assert t.values(sigma)[0] == pytest.approx(expected_value(g, UniformPolicy(g)), abs=1e-12)


def test_tree_advantages_are_counterfactual_gains():
    """A player's advantage-weighted strategy is zero, and a best response has no positive advantage left."""
    from headsup.algos.tabular import CFR

    g = make_game("kuhn")
    t = Tree(g)
    rng = np.random.default_rng(0)
    sigma = np.where(t.info_legal, rng.random(t.info_legal.shape), 0.0)
    sigma /= sigma.sum(1, keepdims=True)
    for p in (0, 1):
        adv = t.advantages(sigma, p)
        mine = t.info_player == p
        np.testing.assert_allclose((adv * sigma).sum(1)[mine], 0.0, atol=1e-12)
        assert not adv[~mine].any() and not adv[~t.info_legal].any()
    eq = CFR(g, "cfr+").iterate(2000).average_policy()
    table = np.stack([eq.table[k] for k in t.info_keys])
    assert max(t.advantages(table, 0).max(), t.advantages(table, 1).max()) < 0.02  # (near-)equilibrium: nothing to gain
