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
    assert ev["average"] < 0.12 and np.isfinite(ev["current"]), ev  # uniform: 0.458; 24 seeds: 0.019-0.067 (dcfr+), 0.014-0.060 (pdcfr+)
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
    """The CUDA-graph path: the replayed baseline fit must see the in-place target copies (a stale target gives an
    error of 0.5), the persistent Adam must count across graphs, and a GPU checkpoint must resume on the GPU."""
    import io

    def steps(solver):
        return [int(next(iter(o.state.values()))["step"]) for o in solver.opt_R]

    s = PDCFRSolver(make_game("kuhn"), "pdcfr+", q_steps=600, q_batch=512, seed=0, device="cuda")
    _fill_baseline_memory(s)
    s._fit_baseline(1)
    exact, legal = _exact_action_values(s, 1)
    assert np.abs((s.q_tab - exact)[legal]).max() < 2e-3
    s = _small("pdcfr+", traversals=500, device="cuda").iterate(3)
    assert np.isfinite(s.evaluate()["average"]) and np.isfinite(s.q_tab).all() and steps(s) == [3 * s.adv_steps] * 2
    buf = io.BytesIO()
    torch.save(s.state_dict(), buf)
    buf.seek(0)
    b = _small("pdcfr+", traversals=500, device="cuda").load_state_dict(torch.load(buf, map_location="cpu", weights_only=False))
    assert all(g["fused"] and g["capturable"] for o in b.opt_R + b.opt_r for g in o.param_groups)
    b.iterate(1)
    assert steps(b) == [4 * b.adv_steps] * 2 and np.isfinite(b.evaluate()["average"])


def test_checkpoint_resume_and_variant_mismatch(tmp_path):
    a = _small("pdcfr+", traversals=300).iterate(2)
    torch.save(a.state_dict(), tmp_path / "ck.pt")
    state = torch.load(tmp_path / "ck.pt", weights_only=False)
    b = _small("pdcfr+", traversals=300).load_state_dict(state)
    assert (b.iteration, b.episodes, b.nodes_touched) == (a.iteration, a.episodes, a.nodes_touched)
    np.testing.assert_array_equal(b.sigma, a.sigma)
    np.testing.assert_array_equal(b.q_tab, a.q_tab)
    assert b.q_memory.size == a.q_memory.size and len(b.strat_memory) == len(a.strat_memory)
    torch.set_rng_state(state["torch_rng"])  # both solvers share torch's global generator in this process
    a.iterate(1)
    torch.set_rng_state(state["torch_rng"])
    b.iterate(1)  # the same random streams: the resumed run continues identically (CPU)
    np.testing.assert_allclose(b.sigma, a.sigma, atol=1e-6)
    with pytest.raises(ValueError, match="variant"):
        _small("dcfr+", traversals=300).load_state_dict(a.state_dict())


def test_cli_writes_a_curve_and_resumes(tmp_path, capsys):
    import json

    from headsup.algos.pdcfr import main

    args = ["--game", "kuhn", "--variant", "dcfr+", "--traversals", "200", "--adv-steps", "20", "--q-steps", "20", "--policy-steps", "50",
            "--checkpoint", str(tmp_path / "ck.pt"), "--checkpoint-minutes", "0", "--json", str(tmp_path / "run.json")]
    main(args + ["--episodes", "1600"])  # 4 iterations
    out = capsys.readouterr().out
    assert "kuhn dcfr+ it 4: exploitability current" in out and "episodes 1600" in out
    curve = json.load(open(tmp_path / "run.json"))["curve"]
    assert [c["iteration"] for c in curve] == [1, 2, 3, 4] and curve[-1]["episodes"] == 1600  # 1, 2, every 3rd, the last
    main(args + ["--episodes", "2400"])  # resumes at iteration 4 and runs to 6
    assert "resumed from" in capsys.readouterr().out
    assert [c["iteration"] for c in json.load(open(tmp_path / "run.json"))["curve"]] == [1, 2, 3, 4, 6]


# ---- the solver's wiring (final review: a mutant of each of these lines survived the tests above) ----------------
def _exact_action_values(s, t):
    """Player 0's exact action values at every decision node under the strategies iteration t + 1 would play."""
    import headsup.algos.pdcfr as P

    tree = s.tree
    v0 = tree.values(s._strategy_table(P.discount(t + 1, s.alpha, s.offset))) / s.scale
    dec = np.flatnonzero(tree.dec >= 0)
    return np.where(tree.legal[dec], v0[np.maximum(tree.child[dec], 0)], 0.0), tree.legal[dec]


def _fill_baseline_memory(s, episodes=20_000):
    tree, rng = s.tree, np.random.default_rng(0)
    mixed = np.where(tree.info_legal, rng.random(tree.info_legal.shape) + 0.2, 0.0)
    mixed /= mixed.sum(1, keepdims=True)
    for p in (0, 1):  # transitions of both traversers under a fully mixed profile
        d = sample_episodes(tree, mixed, s.q_tab, p, episodes, 0.6, rng, s.scale)
        s.q_memory.add(node=d["q_node"], action=d["q_action"], next=d["q_next"], next_info=d["q_next_info"], reward=d["q_reward"], done=d["q_done"])


@pytest.mark.parametrize("variant", ["dcfr+", "pdcfr+"])
def test_fitted_baseline_equals_the_exact_action_values(variant):
    """The whole baseline pipeline at once: the buffer's conventions (player 0's reward, next node and infoset), the
    expected-SARSA target under the NEXT strategy of whoever acts next, and the target-network sync."""
    s = PDCFRSolver(make_game("kuhn"), variant, q_steps=600, q_batch=512, seed=0)
    _fill_baseline_memory(s)
    s._fit_baseline(1)
    exact, legal = _exact_action_values(s, 1)
    assert np.abs((s.q_tab - exact)[legal]).max() < 2e-3  # measured <= 1e-4; a wrong sign / policy / stale target: >= 0.5


@pytest.mark.parametrize("variant", ["dcfr+", "pdcfr+"])
def test_advantage_fit_is_the_authors_loss_on_the_persistent_networks(variant, monkeypatch):
    import copy

    import headsup.algos.pdcfr as P

    s = _small(variant, traversals=300, adv_steps=40, adv_batch=64, q_steps=40, q_batch=64).iterate(2)  # non-trivial networks
    d = discount(3, s.alpha, s.offset)
    data = sample_episodes(s.tree, s._strategy_table(d), s.q_tab, 0, 300, s.epsilon, s.rng, s.scale)
    fixed = torch.arange(64) % len(data["adv_info"])
    got = {}
    monkeypatch.setattr(P, "_batch_index", lambda n, b, dev: fixed)
    monkeypatch.setattr(P, "_optimise", lambda nets, opts, loss_fn, steps, **kw: got.update(nets=nets, opts=opts, loss_fn=loss_fn, steps=steps, kw=kw))
    before = copy.deepcopy(s.R[0])  # the authors' target model: the net as the previous fit left it
    s._fit_advantage(0, data, d)
    x = torch.as_tensor(s.tree.info_obs[data["adv_info"]])[fixed]
    mask = torch.as_tensor(s.tree.info_legal[data["adv_info"]], dtype=torch.float32)[fixed]
    adv = torch.as_tensor(data["adv"], dtype=torch.float32)[fixed]

    def want():
        with torch.no_grad():
            loss = (s.R[0](x) * mask - (torch.clamp(before(x) * mask, min=0.0) * d + adv)).pow(2).mean()
            if variant == "pdcfr+":  # the prediction net: this iteration's advantages only, on the same minibatch
                loss = loss + (s.r[0](x) * mask - adv).pow(2).mean()
        return float(loss)

    assert float(got["loss_fn"]().detach()) == pytest.approx(want(), rel=1e-5)
    with torch.no_grad():  # the target stays the pre-fit network while R moves
        s.R[0][-1].bias.add_(0.3)
    assert float(got["loss_fn"]().detach()) == pytest.approx(want(), rel=1e-5)
    assert got["steps"] == s.adv_steps and got["nets"][0] is s.R[0] and got["opts"][0] is s.opt_R[0]  # never re-created
    if variant == "pdcfr+":
        assert got["nets"][1] is s.r[0] and got["opts"][1] is s.opt_r[0]
    # the "w/o adv" ablation divides the sampled advantages by the traverser's sampling reach
    s.reach_weighted = True
    before = copy.deepcopy(s.R[0])
    s._fit_advantage(0, data, d)
    adv = torch.as_tensor(data["adv"] / data["adv_reach"][:, None], dtype=torch.float32)[fixed]
    assert float(got["loss_fn"]().detach()) == pytest.approx(want(), rel=1e-5)
    assert not np.allclose(data["adv_reach"], 1.0)


def test_iteration_schedule_discounts_alternation_and_baseline_sync(monkeypatch):
    import headsup.algos.pdcfr as P

    s = _small("pdcfr+", traversals=300, adv_steps=60, q_steps=20, policy_steps=20)
    fits, sampled, base = [], [], []
    fit, episodes, optimise = s._fit_advantage, P.sample_episodes, P._optimise

    def spy_fit(p, data, d):
        fits.append((p, d))
        return fit(p, data, d)

    def spy_episodes(tree, sigma, q, traverser, *a, **kw):  # the strategies must come from the networks as they are NOW
        sampled.append(np.array_equal(sigma, s._strategy_table(discount(s.iteration, s.alpha, s.offset))))
        return episodes(tree, sigma, q, traverser, *a, **kw)

    def spy_optimise(nets, opts, loss_fn, steps, **kw):
        if "sync_fn" in kw:
            base.append((steps, kw["sync_every"]))
        return optimise(nets, opts, loss_fn, steps, **kw)

    s._fit_advantage = spy_fit
    monkeypatch.setattr(P, "sample_episodes", spy_episodes)
    monkeypatch.setattr(P, "_optimise", spy_optimise)
    s.iterate(3)
    assert fits == [(p, discount(t, s.alpha, s.offset)) for t in (1, 2, 3) for p in (0, 1)]  # d_t, not d_(t+1)
    assert all(sampled) and len(sampled) == 6  # player 1's episodes already use player 0's updated networks
    assert base == [(s.q_steps, s.q_sync)] * 6 and s.q_sync == 50  # the target net is synced every 50 steps
    assert [int(next(iter(o.state.values()))["step"]) for o in s.opt_R] == [3 * s.adv_steps] * 2  # one Adam per net, for good
    stamps = s.strat_memory.t[: len(s.strat_memory)].cpu().numpy().ravel()
    assert set(stamps.tolist()) == {1.0, 2.0, 3.0}  # strategy samples carry their iteration


@pytest.mark.parametrize("variant", ["dcfr+", "pdcfr+"])
def test_each_player_plays_its_own_networks(variant):
    s = _small(variant)
    A = s.game.num_actions
    bias = np.stack([np.arange(A, 0, -1.0), np.arange(1.0, A + 1)])  # player 0 prefers the first actions, player 1 the last
    with torch.no_grad():  # zero-initialised heads: the output is the bias
        for player in (0, 1):
            s.R[player][-1].bias.copy_(torch.as_tensor(bias[player], dtype=torch.float32))
    sigma = s._strategy_table(0.5)
    want = bias[s.tree.info_player] * s.tree.info_legal
    np.testing.assert_allclose(sigma, want / want.sum(1, keepdims=True), atol=1e-6)
    assert set(s.tree.info_player.tolist()) == {0, 1}


def test_average_policy_weights_the_iterations(monkeypatch):
    """The average-policy net is fitted with weights (2 t / T)^gamma: two iterations with contradictory strategies."""
    s = _small("dcfr+", policy_steps=1500)
    tree, A = s.tree, s.game.num_actions
    first = np.eye(A)[np.argmax(tree.info_legal, axis=1)]
    last = np.eye(A)[A - 1 - np.argmax(tree.info_legal[:, ::-1], axis=1)]
    assert (first != last).any(axis=1).all()
    n = len(first)
    for t, target in ((1, first), (4, last)):
        for _ in range(20):
            s.strat_memory.add(tree.info_obs, np.full(n, t, np.float32), target, tree.info_legal)
    s.iteration = 4
    w1, w4 = (2 * 1 / 4) ** s.gamma, (2 * 4 / 4) ** s.gamma
    want = (w1 * first + w4 * last) / (w1 + w4)  # 0.06 on the iteration-1 action; unweighted: 0.5
    table = s.average_policy().table
    got = np.stack([table[k] for k in tree.info_keys])
    assert np.abs(got - want).max() < 0.05, np.abs(got - want).max()


def test_reinit_prediction_switch_replaces_the_prediction_net():
    quick = dict(traversals=200, adv_steps=10, q_steps=10)
    s = _small("pdcfr+", **quick)
    nets, opts = list(s.r), list(s.opt_r)
    s.iterate(1)
    assert all(a is b for a, b in zip(nets, s.r)) and all(a is b for a, b in zip(opts, s.opt_r))  # default: never re-initialised
    s = _small("pdcfr+", **quick, reinit_prediction=True)
    nets, opts = list(s.r), list(s.opt_r)
    s.iterate(1)
    assert not any(a is b for a, b in zip(nets, s.r)) and not any(a is b for a, b in zip(opts, s.opt_r))


def test_state_dict_is_a_snapshot_and_solvers_never_share_optimiser_state():
    """state_dict() handed out the optimisers' live tensors: a snapshot kept changing, and after an in-memory
    load two solvers stepped each other's Adam."""
    def steps(solver):
        return [int(next(iter(o.state.values()))["step"]) for o in solver.opt_R]

    a = _small("pdcfr+", traversals=200, adv_steps=10, q_steps=10).iterate(1)
    snap = a.state_dict()
    b = _small("pdcfr+", traversals=200, adv_steps=10, q_steps=10).load_state_dict(snap)
    a.iterate(1)
    assert int(snap["opt_R"][0]["state"][0]["step"]) == 10 and steps(a) == [20, 20] and steps(b) == [10, 10]
    b.iterate(2)
    assert steps(a) == [20, 20] and steps(b) == [30, 30]


def test_loading_keeps_this_solvers_optimiser_switches():
    """fused / capturable belong to the device the solver runs on: a checkpoint written on another device type
    must not switch them (a CPU checkpoint left a CUDA solver's persistent Adam non-capturable inside the CUDA graph)."""
    a = _small("pdcfr+", traversals=200, adv_steps=10, q_steps=10).iterate(1)
    state = a.state_dict()
    for group in [g for key in ("opt_R", "opt_r") for o in state[key] for g in o["param_groups"]]:
        group["fused"], group["capturable"] = True, True  # as written by a CUDA solver
    b = _small("pdcfr+", traversals=200, adv_steps=10, q_steps=10)
    fresh = [(g.get("fused"), g.get("capturable")) for o in b.opt_R + b.opt_r for g in o.param_groups]
    b.load_state_dict(state)
    assert [(g.get("fused"), g.get("capturable")) for o in b.opt_R + b.opt_r for g in o.param_groups] == fresh
    b.iterate(1)
