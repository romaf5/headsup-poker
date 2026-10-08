"""Deep (Predictive) Discounted CFR (Xu et al. 2025) on Kuhn / Leduc: pieces and end-to-end convergence."""

import numpy as np
import pytest
import torch

from headsup.algos.best_response import expected_value, exploitability
from headsup.algos.pdcfr import (
    CHANCE,
    DECISION,
    TERMINAL,
    PDCFRSolver,
    Tree,
    advantage_target,
    baseline_target,
    discount,
    mlp,
    sample_episodes,
    strategy_rows,
)
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


def _random_profile(tree, rng, floor=0.25):
    s = np.where(tree.info_legal, rng.random(tree.info_legal.shape), 0.0)
    s /= s.sum(1, keepdims=True)
    return floor * tree.info_legal / tree.info_legal.sum(1, keepdims=True) + (1 - floor) * s


@pytest.mark.parametrize("baseline", [False, True])
def test_sampled_advantages_are_unbiased_with_any_baseline(baseline):
    """E[sampled advantage | infoset visited] is the exact advantage of the current strategy - with no baseline
    and with an arbitrary one (the baseline only changes the variance)."""
    g = make_game("kuhn")
    tree = Tree(g)
    rng = np.random.default_rng(1)
    sigma = _random_profile(tree, rng)
    q = rng.normal(scale=0.5, size=(tree.num_decisions, 3)) if baseline else np.zeros((tree.num_decisions, 3))
    for p in (0, 1):
        data = sample_episodes(tree, sigma, q, p, 400_000, 0.6, rng, scale=tree.max_utility)
        total = np.zeros((tree.num_infosets, 3))
        np.add.at(total, data["adv_info"], data["adv"])
        count = np.bincount(data["adv_info"], minlength=tree.num_infosets)
        mine = tree.info_player == p
        assert count[mine].min() > 5000 and not count[~mine].any()
        exact = tree.advantages(sigma, p) / tree.max_utility
        np.testing.assert_allclose(total[mine] / count[mine][:, None], exact[mine], atol=0.03)
        assert not data["adv"][~tree.info_legal[data["adv_info"]]].any()  # illegal actions carry no advantage
        assert (tree.info_player[data["strat_info"]] == 1 - p).all()


def test_sampler_transitions_and_counts():
    g = make_game("leduc")
    tree = Tree(g)
    rng = np.random.default_rng(0)
    sigma = _random_profile(tree, rng)
    n = 2000
    data = sample_episodes(tree, sigma, np.zeros((tree.num_decisions, 3)), 0, n, 0.6, rng, scale=13.0)
    j = len(data["q_node"])
    assert j == len(data["adv_info"]) + len(data["strat_info"])  # one transition per decision
    assert int(data["q_done"].sum()) == n  # every episode ends exactly once
    done = data["q_done"] > 0
    assert not data["q_reward"][~done].any() and np.abs(data["q_reward"]).max() <= 1.0
    assert (data["q_next"][done] == 0).all() and (data["q_next_info"][done] == 0).all()
    assert tree.legal[np.flatnonzero(tree.dec >= 0)[data["q_node"]], data["q_action"]].all()  # only legal actions are taken
    chance_per_episode = 3  # two private cards, one public card (when round 2 is reached)
    assert n * 2 + j + n <= data["nodes"] <= n * chance_per_episode + j + n
    assert (data["adv_reach"] > 0).all() and (data["adv_reach"] <= 1.0).all()


def test_sampler_never_takes_a_zero_probability_action():
    """One-hot strategies (iteration 1: every network outputs zero) and no exploration: every sampled action has
    probability one, values stay finite."""
    g = make_game("leduc")
    tree = Tree(g)
    last = tree.num_actions - 1 - np.argmax(tree.info_legal[:, ::-1], axis=1)  # the last legal action of each infoset
    sigma = np.zeros(tree.info_legal.shape)
    sigma[np.arange(tree.num_infosets), last] = 1.0
    data = sample_episodes(tree, sigma, np.zeros((tree.num_decisions, 3)), 1, 20_000, 0.0, np.random.default_rng(3), scale=13.0)
    node = np.flatnonzero(tree.dec >= 0)[data["q_node"]]
    assert (data["q_action"] == last[tree.info[node]]).all()
    assert np.isfinite(data["adv"]).all() and np.abs(data["adv"]).max() <= 2.0


def test_exact_baseline_removes_most_sampling_variance_for_both_players():
    """The baseline is PLAYER 0's value and is negated for player 1: with the exact action values of the current
    profile as baseline, the sampled advantages on Kuhn (no chance below the deal) lose most of their variance
    for either traverser (an unbiased estimator with a wrong-signed baseline would gain variance instead)."""
    g = make_game("kuhn")
    tree = Tree(g)
    rng = np.random.default_rng(2)
    sigma = _random_profile(tree, rng)
    v0 = tree.values(sigma) / tree.max_utility
    dec = np.flatnonzero(tree.dec >= 0)
    exact = np.where(tree.legal[dec], v0[np.maximum(tree.child[dec], 0)], 0.0)  # (num_decisions, A): player 0's action values
    for p in (0, 1):
        var = []
        for q in (np.zeros_like(exact), exact):
            data = sample_episodes(tree, sigma, q, p, 100_000, 0.6, rng, scale=tree.max_utility)
            mean = np.zeros((tree.num_infosets, 3))
            np.add.at(mean, data["adv_info"], data["adv"])
            mean /= np.maximum(np.bincount(data["adv_info"], minlength=tree.num_infosets), 1)[:, None]
            var.append(((data["adv"] - mean[data["adv_info"]]) ** 2).mean())
        assert var[1] < 0.25 * var[0], (p, var)


def test_advantage_target_clips_the_previous_output_at_read_time():
    frozen = torch.tensor([[2.0, -3.0, 5.0]])
    legal = torch.tensor([[1.0, 1.0, 0.0]])
    adv = torch.tensor([[0.5, -0.25, 0.0]])
    torch.testing.assert_close(advantage_target(frozen, legal, 0.5, adv), torch.tensor([[1.5, -0.25, 0.0]]))
    torch.testing.assert_close(advantage_target(frozen, legal, 0.0, adv), adv)  # iteration 1: d = 0


def test_baseline_target_is_expected_sarsa():
    reward = torch.tensor([0.0, -1.0])
    done = torch.tensor([0.0, 1.0])
    next_q = torch.tensor([[1.0, 3.0, 100.0], [7.0, 7.0, 7.0]])
    next_sigma = torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(baseline_target(reward, done, next_q, next_sigma), torch.tensor([2.5, -1.0]))


def _small(variant, **kw):
    args = dict(traversals=1000, adv_steps=100, adv_batch=256, q_steps=100, q_batch=256, policy_steps=400, policy_batch=256, seed=0)
    args.update(kw)
    return PDCFRSolver(make_game("kuhn"), variant, **args)


@pytest.mark.parametrize("variant", ["dcfr+", "pdcfr+"])
def test_kuhn_converges(variant):
    s = _small(variant)
    assert (s.alpha, s.gamma, s.offset) == ((2.0, 2.0, 1.5) if variant == "dcfr+" else (2.3, 2.0, 1.0))
    s.iterate(15)
    ev = s.evaluate()
    assert ev["average"] < 0.09 and np.isfinite(ev["current"]), ev  # uniform: 0.458; measured 0.019 / 0.042 (0.018 / 0.020 at 30 iterations)
    assert s.iteration == 15 and s.episodes == 15 * 2 * 1000 and s.nodes_touched > s.episodes * 3
    pol = s.average_policy()
    for probs in pol.table.values():
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert (s.r is None) == (variant == "dcfr+")


def test_ablation_switches_run():
    quick = dict(traversals=300, adv_steps=30, q_steps=30, policy_steps=50)
    for kw in (dict(baseline=False), dict(reach_weighted=True), dict(reinit_prediction=True), dict(fallback="uniform")):
        s = _small("pdcfr+", **quick, **kw).iterate(3)
        assert np.isfinite(s.evaluate()["average"])
    assert not _small("pdcfr+", **quick, baseline=False).iterate(2).q_tab.any()


def test_tiny_budget_and_untrained_solver():
    g = make_game("kuhn")
    s = PDCFRSolver(g, "pdcfr+", traversals=5, adv_steps=5, q_steps=5, policy_steps=5, seed=0)
    assert s.evaluate()["average"] == pytest.approx(exploitability(g, UniformPolicy(g))[0], abs=1e-9)  # untrained: uniform
    s.iterate(3)  # far fewer samples than the 2 048 minibatch
    assert np.isfinite(s.evaluate()["average"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda():
    s = _small("pdcfr+", traversals=500, device="cuda").iterate(3)
    assert np.isfinite(s.evaluate()["average"]) and np.isfinite(s.q_tab).all()
