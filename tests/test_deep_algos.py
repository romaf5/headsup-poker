"""Deep CFR / SD-CFR / DREAM / ESCHER on the game protocol (Kuhn), against exact exploitability."""

import numpy as np
import pytest
import torch

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


def test_legal_mask_from_history_matches_the_game():
    """The DREAM baselines' dueling head masks with the actor's legal actions, recomputed from the two-player history input."""
    import numpy as np
    import torch

    from headsup.games.leduc import legal_mask_from_history

    g = make_game("leduc")
    rows, masks = [], []

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        rows.append(np.concatenate([s.info_state(0), s.info_state(1)]))
        masks.append(s.legal_mask())
        for a in s.legal_actions():
            walk(s.child(a))

    walk(g.new_initial_state())
    got = legal_mask_from_history(g, torch.as_tensor(np.stack(rows)), g.obs_dim).numpy() > 0
    np.testing.assert_array_equal(got, np.stack(masks))
    for in_dim in (g.obs_dim, 2 * g.obs_dim):
        m = g.make_model(arch="deepcfr_dueling", in_dim=in_dim)
        x = torch.as_tensor(np.stack(rows)[:, :in_dim])
        legal = torch.as_tensor(np.stack(masks))
        with torch.no_grad():
            assert torch.all(m(x)[~legal] == 0)


def test_dream_bootstrap_chance_only_changes_the_baseline_targets():
    """Same seed, same nets: the deal is sampled in the same order, so trajectories, regrets and node counts are identical;
    only the expected-SARSA targets of transitions into a deal change (bootstrapped after the deal instead of sampled)."""
    import numpy as np

    g = make_game("leduc")
    out = {}
    for flag in (False, True):
        s = DeepSolver(g, "dream", traversals=50, adv_steps=5, q_steps=5, seed=0, bootstrap_chance=flag)
        s.rng = np.random.default_rng(7)
        adv, q = [], []
        for _ in range(200):
            s._os_dream(g.new_initial_state(), 0, 1, 1.0, adv, q)
        out[flag] = (adv, q, s.nodes_touched)
    (a0, q0, n0), (a1, q1, n1) = out[False], out[True]
    assert n0 == n1 and len(a0) == len(a1) and len(q0) == len(q1)
    for x, y in zip(a0, a1):
        np.testing.assert_array_equal(x[2], y[2])
    changed = sum(not np.array_equal(x[1], y[1]) for x, y in zip(q0, q1))
    assert 0 < changed < len(q0)  # only the round-1 transitions that end the round


def test_dream_shared_baseline_is_zero_sum_and_resumes(tmp_path):
    import numpy as np
    import torch

    g = make_game("leduc")
    s = DeepSolver(g, "dream", traversals=30, adv_steps=5, q_steps=20, seed=0, shared_baseline=True, bootstrap_chance=True)
    s.iterate()
    assert len(s.q_nets) == 1 and s.q_memory[0].size > 0
    state = g.new_initial_state()
    while state.is_chance():
        state = state.child(state.sample_chance(np.random.default_rng(0)))
    np.testing.assert_allclose(s._q(1, state), -s._q(0, state))
    torch.save(s.state_dict(), tmp_path / "ck.pt")
    t = DeepSolver(g, "dream", traversals=30, adv_steps=5, q_steps=20, seed=0, shared_baseline=True, bootstrap_chance=True)
    t.load_state_dict(torch.load(tmp_path / "ck.pt", weights_only=False))
    np.testing.assert_allclose(t._q(0, state), s._q(0, state))
    t.iterate()
    assert t.iteration == 2


def test_dream_checkpoint_baseline_layout_must_match():

    g = make_game("kuhn")
    s = DeepSolver(g, "dream", traversals=10, adv_steps=2, q_steps=2, seed=0, shared_baseline=True)
    s.iterate()
    with pytest.raises(ValueError, match="shared-baseline"):
        DeepSolver(g, "dream", traversals=10, adv_steps=2, q_steps=2, seed=0).load_state_dict(s.state_dict())


def test_leduc_report_rows_and_unknown_algorithms(tmp_path):
    import json

    from headsup.algos.leduc_report import _parse_row, table

    assert _parse_row("dreamp,dream,DREAM, mine (v2),runs/x/*.json") == ("dreamp", "dream", "DREAM, mine (v2)", "runs/x/*.json")
    (tmp_path / "e_s0.json").write_text(json.dumps({"curve": [{"iteration": 10, "average": 0.1, "nodes_touched": 1e6}]}))
    with pytest.raises(ValueError, match="escher"):
        table([("dreamp", "escher", "ESCHER", str(tmp_path / "e_s*.json"))], "dreamp", (1e6,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimise_steps_several_networks_and_syncs(device):
    import torch

    from headsup.algos.deep import _adam, _optimise

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    a, b = torch.nn.Linear(3, 1).to(device), torch.nn.Linear(3, 1).to(device)
    x = torch.randn(64, 3, device=device)
    ya, yb = x.sum(1, keepdim=True), -x.sum(1, keepdim=True)
    calls = []
    loss = _optimise([a, b], [_adam(a, 3e-2), _adam(b, 3e-2)], lambda: (a(x) - ya).pow(2).mean() + (b(x) - yb).pow(2).mean(),
                     300, grad_clip=0, sync_every=50, sync_fn=lambda: calls.append(1))
    assert loss < 0.05  # both regressions were optimised
    assert len(calls) == 6  # after steps 1, 51, 101, 151, 201, 251


def test_leduc_report_tabulates_pdcfr_runs_by_episodes(tmp_path):
    import json

    from headsup.algos.leduc_report import PDCFR_PAPER, load_run, table

    curve = [{"iteration": i, "average": 0.2 - 0.0002 * i, "current": 1.0, "nodes_touched": 150_000 * i, "episodes": 20_000 * i}
             for i in range(1, 501)]
    (tmp_path / "x_s0.json").write_text(json.dumps({"curve": curve}))
    assert load_run(str(tmp_path / "x_s0.json"))[49] == (50, pytest.approx(190.0), 7_500_000, 1_000_000)
    out = table([("pdcfrp", "pdcfr+", "ours", str(tmp_path / "x_s*.json"))], "pdcfrp", (1e6, 4e6, 9.5e6))
    lines = out.splitlines()
    assert lines[0].startswith("| episodes |") and "9.5e6" in lines[0]
    assert lines[2] == "| **VR-DeepPDCFR+, paper** | **158** | **115** | **90** |"
    assert lines[3].startswith("| ours (1) | 190 | 160 | 105 |")  # means within +-10 % (the last column: 9-10 M) of x
    assert PDCFR_PAPER[("pdcfrk", "dcfr+")][9.5e6] == pytest.approx(5.3)


def test_leduc_report_tabulates_rebel_runs_and_labels_its_references(tmp_path):
    import json

    from headsup.algos.leduc_report import REBEL_EXACT_LEAVES, REBEL_FULL_LCFR, rebel_references, rebel_table

    for seed, shift in ((0, 0.0), (1, 0.002)):
        curve = [{"epoch": e, "examples": 7000 * e, "sgd_steps": 50 * e, "games": 2048 * e, "exploitability": 0.5 / e + shift,
                  "exploitability_sampled": 0.6 / e + shift, "samples": 1024, "exploitability_unsafe": 0.7 / e, "root_value": -0.08,
                  "value_error": 1.0 / e, "value_error_search": None, "loss": 0.1, "seconds": 3.0 * e} for e in (1, 25, 50, 75, 100)]
        (tmp_path / f"leduc_s{seed}.json").write_text(json.dumps({"game": "leduc", "algo": "rebel", "curve": curve[: 5 - seed]}))
    lines = rebel_table(str(tmp_path / "leduc_s*.json"), epochs=(1, 50, 100)).splitlines()
    assert lines[0] == "| epoch (2 runs) | 1 | 50 | 100 | last 3 evaluations (epochs 25 / 50 / 75) |"
    # mA/g; the last column: each run's mean over the last three evaluations that EVERY run has (seed 1 stops at 75)
    tail = [1000 * np.mean([0.5 / e + shift for e in (25, 50, 75)]) for shift in (0.0, 0.002)]
    assert lines[2] == ("| policy played in expectation (exact mixture over the stopping steps) | 501.0 ± 1.4 | 11.0 ± 1.4 | 5.0 | "
                        f"{np.mean(tail):.1f} ± {np.std(tail, ddof=1):.1f} |")
    assert lines[3].startswith("| average of K = 1024 sampled playthrough policies (the paper's protocol) | 601.0 ± 1.4 |")
    assert "| 700.0 ± 0.0 | 14.0 ± 0.0 | 7.0 |" in lines[4] and "unsafe" in lines[4]
    assert lines[6].endswith("| – | – | – | – |")  # a quantity the runs do not have
    assert lines[8].startswith("| training examples generated, thousands | 7.0 ± 0.0 | 350.0 ± 0.0 | 700.0 |")
    head = rebel_table(str(tmp_path / "leduc_s*.json")).splitlines()[0]
    assert head.startswith("| epoch (2 runs) | 1 | 25 | 50 | 75 | last 3")  # the epochs every run has
    assert rebel_table(str(tmp_path / "none_s*.json")).startswith("(no runs match")
    oracle = {"curve": [{"iters": 1024, "exploitability": 0.0221, "exploitability_unsafe": 0.0545}]}
    (tmp_path / "oracle_T1024.json").write_text(json.dumps(oracle))
    ref = rebel_references(str(tmp_path / "oracle_T*.json")).splitlines()
    assert ref[0] == "| search steps T (T / 2 updates per player) | 64 | 128 | 256 | 512 | 1024 | 2048 |"
    assert ref[2] == "| full-game tabular Linear CFR, average strategy (ours) | 139.5 | 55.6 | 41.0 | 16.2 | 10.6 | 5.2 |"
    assert ref[3] == ("| search with exact leaf values, random-iterate mixture (the brief's prototype) "
                      "| 134.9 | 80.1 | 49.9 | 32.7 | 21.8 | 14.0 |")
    assert ref[5] == "| search with exact leaf values, random-iterate mixture (ours, `--oracle`) | – | – | – | – | 22.1 | – |"
    assert ref[6].endswith("| – | – | – | – | 54.5 | – |")
    pinouche = [line for line in ref if line.startswith("| pinouche/poker_self_play")]
    assert "third party" in pinouche[0] and pinouche[0].endswith("| 58.5 | 46.6 | 26.9 | 16.4 |")
    assert pinouche[1].endswith("| 57.1 | 49.7 | 39.2 | 29.3 |")
    assert ref[-1].startswith("| Student of Games, Fig. 3A") and ref[-1].endswith("| ~200 | ~105 | ~55 | ~22 |")
    assert REBEL_EXACT_LEAVES[1024] == (21.8, 53.5) and REBEL_FULL_LCFR[1024] == 10.6


def test_current_policy_is_the_strategy_the_solver_plays():
    """current_policy() used the uniform fallback while the traversals use the solver's (argmax by default): the
    'current' exploitability column described a strategy that was never played."""
    g = make_game("leduc")
    for argmax in (True, False):
        s = DeepSolver(g, "sdcfr", traversals=100, adv_steps=30, adv_batch=128, seed=0, rm_argmax=argmax)
        s.iterate(2)
        pol = s.current_policy()
        differ, fallback = 0, 0
        stack = [g.new_initial_state()]
        while stack:
            st = stack.pop()
            if st.is_terminal():
                continue
            if st.is_chance():
                stack.extend(st.child(a) for a, _ in st.chance_outcomes())
                continue
            played = s._sigma(st.current_player, st)
            differ += not np.allclose(pol(st), played, atol=1e-6)  # float32 noise of batched vs single forwards: < 1e-7
            fallback += played.max() == 1.0
            stack.extend(st.child(a) for a in st.legal_actions())
        assert differ == 0 and (fallback > 0 or not argmax)  # with argmax, some infosets do use the fallback


def test_escher_counts_terminal_nodes_like_the_other_samplers():
    """nodes_touched omitted ESCHER's terminal states (external sampling and DREAM count them)."""
    s = DeepSolver(make_game("kuhn"), "escher", traversals=50, adv_steps=2, policy_steps=2, q_steps=2, seed=0)
    visited = [0]

    class Counting:  # a state that counts every state a trajectory visits, the terminal one included
        def __init__(self, state):
            self._s = state
            visited[0] += 1

        def child(self, a):
            return Counting(self._s.child(a))

        def __getattr__(self, name):
            return getattr(self._s, name)

    class CountingGame:
        def __init__(self, game):
            self._g = game

        def new_initial_state(self):
            return Counting(self._g.new_initial_state())

        def __getattr__(self, name):
            return getattr(self._g, name)

    s.game = CountingGame(s.game)
    s.iterate(1)
    assert s.nodes_touched == visited[0]


@pytest.mark.parametrize("algo", ["sdcfr", "deepcfr"])
def test_evaluation_does_not_change_the_training_trajectory(algo):
    """evaluate() consumed the training generator (a fresh net's initialisation, the policy fit's minibatches): the
    run depended on --eval-every."""
    def run(evaluate):
        s = DeepSolver(make_game("kuhn"), algo, traversals=50, adv_steps=20, adv_batch=64, policy_steps=20, policy_batch=64, seed=0)
        s.iterate(2)
        if evaluate:
            s.evaluate()
        s.iterate(1)
        return torch.cat([p.detach().flatten() for n in s.nets for p in n.parameters()])

    torch.testing.assert_close(run(True), run(False))


def test_solver_state_dict_is_a_snapshot():
    """state_dict() handed out the baseline optimisers' live tensors (and the live list of iterates)."""
    s = DeepSolver(make_game("kuhn"), "dream", traversals=30, adv_steps=5, q_steps=5, seed=0).iterate(1)
    snap = s.state_dict()
    step = int(snap["q_opts"][0]["state"][0]["step"])
    n_iterates = len(snap["iterates"][0])
    s.iterate(1)
    assert int(snap["q_opts"][0]["state"][0]["step"]) == step and len(snap["iterates"][0]) == n_iterates


def test_escher_preset_is_the_papers_table_3():
    from headsup.algos.deep import parse_args

    a = parse_args(["--algo", "escher", "--preset", "escher", "--q-batch", "256"])
    assert (a.traversals, a.value_traversals, a.adv_steps, a.adv_batch, a.q_steps, a.q_batch, a.policy_steps, a.policy_batch) == (
        1000, 1000, 5000, 2048, 5000, 256, 10000, 2048)
    assert parse_args(["--algo", "escher"]).q_steps == 1000  # without the preset: the defaults


def test_options_without_effect_for_the_algorithm_are_reported(capsys):
    """--mean-regret with DREAM, --epsilon with SD-CFR, ... were accepted and silently ignored."""
    from headsup.algos.deep import parse_args

    parse_args(["--algo", "dream", "--mean-regret", "--policy-steps", "10", "--epsilon", "0.4", "--q-steps", "5"])
    err = capsys.readouterr().err
    assert "--mean-regret" in err and "--policy-steps" in err and "--epsilon" not in err and "--q-steps" not in err
    parse_args(["--algo", "sdcfr", "--epsilon", "0.4", "--shared-baseline", "--value-traversals", "9", "--mean-regret"])
    err = capsys.readouterr().err
    assert all(flag in err for flag in ("--epsilon", "--shared-baseline", "--value-traversals")) and "--mean-regret" not in err
    parse_args(["--algo", "escher", "--preset", "escher"])
    assert capsys.readouterr().err == ""  # a preset's values are not "given"
