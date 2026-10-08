"""Neural Fictitious Self-Play (Heinrich & Silver 2016) on Kuhn / Leduc: the formulas, the hand-written networks
against torch, the wiring of play and learning, checkpoints, and end-to-end convergence."""

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from headsup.algos.best_response import TabularPolicy, best_response, exploitability
from headsup.algos.nfsp import PRESETS, Memory, NFSPSolver, cross_entropy, epsilon, make_net, td_target
from headsup.algos.pdcfr import CHANCE, DECISION, TERMINAL, Tree
from headsup.games import make_game

_TREES = {}


def _tree(name):
    if name not in _TREES:
        _TREES[name] = Tree(make_game(name))
    return _TREES[name]


def _solver(preset="paper", game="kuhn", **kw):
    return NFSPSolver(make_game(game), preset, tree=_tree(game), **kw)


# ---- the formulas ------------------------------------------------------------------------------------------------
def test_epsilon_schedules():
    assert epsilon(1, 0.06) == pytest.approx(0.06)  # the paper: proportional to 1 / sqrt(iteration)
    assert epsilon(4, 0.06) == pytest.approx(0.03) and epsilon(10_000, 0.06) == pytest.approx(0.0006)
    assert epsilon(1, 0.06, 0.01) == pytest.approx(0.06)  # the DREAM code: start / (1 + const sqrt(completed iterations))
    assert epsilon(10_001, 0.06, 0.01) == pytest.approx(0.03) and epsilon(1_000_001, 0.06, 0.01) == pytest.approx(0.06 / 11)
    assert epsilon(500, 0.3, 0.0) == pytest.approx(0.3)  # no decay


def test_td_target_masks_the_max_and_stops_at_terminals():
    reward = np.array([0.0, 0.0, -2.0, 1.5], np.float32)
    done = np.array([0.0, 0.0, 1.0, 1.0], np.float32)
    target = np.array([[1.0, 3.0, 9.0], [-4.0, -2.0, -3.0], [5.0, 5.0, 5.0], [7.0, 8.0, 9.0]], np.float32)
    legal = np.array([[True, True, False], [False, True, True], [True, True, True], [False, True, False]])
    y = td_target(reward, done, target, legal)
    np.testing.assert_allclose(y, [3.0, -2.0, -2.0, 1.5])  # the illegal 9 is ignored; nothing is added at the end of a hand
    assert y.dtype == np.float32
    # Double DQN: the online network picks among the legal actions, the target network values the pick
    online = np.array([[5.0, 0.0, 99.0], [50.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32)
    np.testing.assert_allclose(td_target(reward, done, target, legal, online), [1.0, -3.0, -2.0, 1.5])


def test_cross_entropy_is_torch_cross_entropy_over_the_legal_logits():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(64, 3)).astype(np.float32) * 3
    legal = rng.random((64, 3)) < 0.6
    action = rng.integers(0, 3, 64)
    legal[np.arange(64), action] = True  # the taken action is legal
    loss, grad = cross_entropy(logits, legal, action.astype(np.int8))
    t = torch.tensor(logits, requires_grad=True)
    want = F.cross_entropy(torch.where(torch.as_tensor(legal), t, torch.full_like(t, -1e20)), torch.as_tensor(action))
    want.backward()
    assert loss == pytest.approx(float(want.detach()), rel=1e-5)
    np.testing.assert_allclose(grad, t.grad.numpy(), atol=1e-7)
    assert grad.dtype == np.float32 and not grad[~legal].any()  # illegal logits get no gradient
    unmasked = F.cross_entropy(torch.tensor(logits), torch.as_tensor(action))
    assert (legal.sum(1) < 3).any() and abs(float(want.detach()) - float(unmasked)) > 0.1  # the mask matters here


# ---- memories ----------------------------------------------------------------------------------------------------
_FIELDS = (("info", np.int32), ("action", np.int8))


def test_circular_memory_keeps_the_latest_rows():
    m = Memory(5, _FIELDS, np.random.default_rng(0))
    m.add(info=np.arange(3), action=np.array([0, 1, 2]))
    assert m.size == 3 and m.data["info"][:3].tolist() == [0, 1, 2] and m.data["action"].dtype == np.int8
    m.add(info=np.arange(3, 7), action=np.array([0, 1, 2, 0]))
    assert m.size == 5 and sorted(m.data["info"].tolist()) == [2, 3, 4, 5, 6]  # a FIFO: 0 and 1 were overwritten
    m.add(info=np.zeros(0, np.int64), action=np.zeros(0, np.int64))  # nothing to add
    m.add(info=np.array([7]), action=1)  # scalars are broadcast
    assert sorted(m.data["info"].tolist()) == [3, 4, 5, 6, 7] and m.seen == 8


def test_reservoir_is_a_uniform_sample_of_everything_offered():
    counts, n, cap = np.zeros(1000), 400, 100
    rng = np.random.default_rng(1)
    for _ in range(n):
        m = Memory(cap, _FIELDS, rng, reservoir=True)
        for start in range(0, 1000, 125):  # several rows per call, as the solver adds them
            m.add(info=np.arange(start, start + 125), action=0)
        assert m.size == cap and m.seen == 1000 and len(set(m.data["info"].tolist())) == cap
        counts[m.data["info"]] += 1
    p = counts / n  # every item is kept with probability capacity / seen = 0.1 - the first as often as the last
    assert abs(p[:100].mean() - 0.1) < 0.01 and abs(p[-100:].mean() - 0.1) < 0.01 and abs(p[100:900].mean() - 0.1) < 0.005
    assert p.min() > 0.03 and p.max() < 0.2


def test_reservoir_minimum_probability_and_window_favour_recent_rows():
    rng = np.random.default_rng(2)
    plain, floor, window = (Memory(100, _FIELDS, rng, reservoir=True), Memory(100, _FIELDS, rng, reservoir=True, min_prob=0.25),
                            Memory(100, _FIELDS, rng))
    for start in range(0, 4000, 50):
        for m in (plain, floor, window):
            m.add(info=np.arange(start, start + 50), action=0)
    recent = [float((m.data["info"] >= 3600).mean()) for m in (plain, floor, window)]  # the last 10 % of the stream
    assert recent[0] < 0.25 and 0.4 < recent[1] < 0.9 and recent[2] == 1.0  # exponential averaging: 1 - 0.9975^400 = 0.63


def test_memory_snapshot_round_trip():
    rng = np.random.default_rng(3)
    m = Memory(50, _FIELDS, rng, reservoir=True)
    m.add(info=np.arange(80), action=1)
    snap = m.state_dict()
    kept = copy.deepcopy(snap)
    other = Memory(50, _FIELDS, np.random.default_rng(4), reservoir=True).load_state_dict(snap)
    m.add(info=np.arange(80, 300), action=2)  # neither the snapshot nor the copy follows the source
    assert (other.size, other.seen) == (50, 80) and np.array_equal(other.data["info"], kept["info"])
    assert all(np.array_equal(snap[k], kept[k]) for k in ("info", "action")) and snap["seen"] == 80
    other.add(info=np.arange(5), action=0)
    assert np.array_equal(snap["info"], kept["info"])


# ---- the networks: numpy forward / backward against the torch modules -----------------------------------------------
_NETS = [("mlp", False), ("mlp", True), ("deepcfr_dueling", False), ("deepcfr_dueling", True)]


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
@pytest.mark.parametrize("arch,policy", _NETS)
def test_network_forward_equals_the_torch_module(game, arch, policy):
    g, tree = make_game(game), _tree(game)
    torch.manual_seed(0)
    net = make_net(g, arch, 64, 1, policy)
    module = net.torch_module()
    with torch.no_grad():
        want = module(torch.as_tensor(tree.info_obs)).numpy()
    got = net.forward(tree.info_obs, tree.info_legal)
    assert got.dtype == np.float32 and got.shape == (tree.num_infosets, g.num_actions)
    np.testing.assert_allclose(got, want, atol=2e-5)
    if arch == "mlp":
        assert [tuple(p.shape) for p in module.parameters()] == [(64, g.obs_dim), (64,), (g.num_actions, 64), (g.num_actions,)]
        assert float(module[-1].weight.detach().abs().min()) > 0  # PyTorch's default initialisation: no zero output layer
    else:  # the repository's DREAM-style net: illegal outputs are exactly 0 (Q) / -1e20 (policy logits)
        reference = g.make_model(hidden=64, arch="deepcfr_dueling", policy=policy)
        assert [k for k, _ in module.named_parameters()] == [k for k, _ in reference.named_parameters()]
        assert (got[~tree.info_legal] == (-1e20 if policy else 0.0)).all()


def test_mlp_depth_and_width():
    g = make_game("kuhn")
    net = make_net(g, "mlp", 16, 3, False)
    assert [tuple(v.shape) for v in net.w.values()] == [(16, 22), (16,), (16, 16), (16,), (16, 16), (16,), (3, 16), (3,)]
    with pytest.raises(ValueError, match="arch"):
        make_net(g, "resnet", 64, 1, False)


@pytest.mark.parametrize("arch,policy", _NETS)
@pytest.mark.parametrize("clip", [0.0, 0.05, 1000.0])  # none, far below the gradient norm, far above it
def test_network_step_is_sgd_with_gradient_norm_clipping(arch, policy, clip):
    """backward() + step() against torch autograd + clip_grad_norm_ + torch.optim.SGD, for an arbitrary loss gradient."""
    g, tree = make_game("leduc"), _tree("leduc")
    torch.manual_seed(1)
    net = make_net(g, arch, 64, 2, policy)
    module = net.torch_module()
    rng = np.random.default_rng(0)
    rows = rng.integers(0, tree.num_infosets, 48)
    x, legal = tree.info_obs[rows], tree.info_legal[rows]
    coef = np.where(legal, rng.normal(size=legal.shape), 0.0).astype(np.float32)  # d loss / d output, at the legal outputs
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    out = module(torch.as_tensor(x))
    (torch.where(torch.as_tensor(legal), out, torch.zeros_like(out)) * torch.as_tensor(coef)).sum().backward()
    norm = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in module.parameters())))
    if clip:
        torch.nn.utils.clip_grad_norm_(module.parameters(), clip)
    opt.step()
    net.forward(x, legal)
    net.backward(coef)
    assert net.step(0.1, clip) == pytest.approx(norm, rel=1e-4) and norm > 1.0  # returns the gradient norm before clipping
    for k, p in module.named_parameters():
        np.testing.assert_allclose(net.w[k], p.detach().numpy(), atol=3e-6, err_msg=k)
    assert net.torch_module() is not module and net.w.keys() == dict(module.named_parameters()).keys()


def test_network_state_is_copied_in_and_out():
    g = make_game("kuhn")
    a, b = make_net(g, "mlp", 8, 1, False), make_net(g, "mlp", 8, 1, False)
    state = a.state_dict()
    b.load_state_dict(state)
    a.grad[:] = 1.0
    a.step(0.5)
    assert all(np.array_equal(b.w[k], state[k]) for k in state) and not np.array_equal(a.w["0.bias"], state["0.bias"])
    b.grad[:] = 1.0
    b.step(0.25)
    assert not np.array_equal(b.w["0.bias"], state["0.bias"]) and np.allclose(a.w["0.bias"] + 0.25, b.w["0.bias"])


def test_dueling_normalisation_survives_a_constant_layer():
    """All trunk units dead: torch's std has a NaN gradient there, the hand-written one is zero."""
    g, tree = make_game("kuhn"), _tree("kuhn")
    net = make_net(g, "deepcfr_dueling", 16, 1, False)
    for k in ("trunk.0.weight", "trunk.1.weight", "trunk.2.weight"):
        net.w[k][:] = 0.0
    for k in ("trunk.0.bias", "trunk.1.bias", "trunk.2.bias"):
        net.w[k][:] = -1.0
    out = net.forward(tree.info_obs, tree.info_legal)
    net.backward(np.where(tree.info_legal, 1.0, 0.0).astype(np.float32))
    assert np.isfinite(out).all() and np.isfinite(net.grad).all()


# ---- play: acting, modes, memories ---------------------------------------------------------------------------------
def _constant(net, values):
    """Make an MLP output ``values`` at every input."""
    weight, bias = list(net.w)[-2:]
    net.w[weight][:] = 0.0
    net.w[bias][:] = values


def _spy_actions(s):
    calls, act = [], s._act

    def spy(info, br, eps):
        calls.append((act(info, br, eps), eps))
        return calls[-1][0]

    s._act = spy
    return calls


@pytest.mark.parametrize("game,widths", [("kuhn", [2, 6]), ("leduc", [4, 5, 30])])
def test_deals_follow_the_chance_probabilities(game, widths):
    """A new hand is ONE draw over all deals (the two private cards are two chance nodes in a row) and a board card
    one draw over the remaining cards - each outcome with the game's probability."""
    from headsup.algos.nfsp import chance_closure

    s = _solver("paper", game=game)
    tree = s.tree
    row, outcome, cum = chance_closure(tree)
    chance = np.flatnonzero(tree.kind == CHANCE)
    assert (row[chance] >= 0).all() and (row[tree.kind != CHANCE] == -1).all() and len(set(row[chance].tolist())) == len(chance)
    seen = set()
    for c in chance:
        width = int((cum[row[c]] <= 1.0).sum())
        nodes, probs = outcome[row[c], :width], np.diff(cum[row[c], :width], prepend=0.0)
        assert (tree.kind[nodes] != CHANCE).all() and len(set(nodes.tolist())) == width and cum[row[c], width - 1] == 1.0
        np.testing.assert_allclose(probs, 1.0 / width, atol=1e-12)  # every deal of these games is uniform
        seen.add(width)
    deals = widths[-1]
    assert sorted(seen) == widths and int((cum[row[0]] <= 1.0).sum()) == deals  # second card, [board card,] both private cards
    start = s._deal(np.zeros(60_000, dtype=np.int64))
    nodes, counts = np.unique(start, return_counts=True)
    assert len(nodes) == deals and (tree.kind[nodes] == DECISION).all() and (tree.player[nodes] == 0).all()
    assert np.abs(counts - 60_000 / deals).max() < 5 * (60_000 / deals) ** 0.5  # uniform over the deals
    if game == "leduc":
        c = int(next(i for i in chance if i > 0 and int((cum[row[i]] <= 1.0).sum()) == 4))  # a board card
        nodes, counts = np.unique(s._deal(np.full(20_000, c, dtype=np.int64)), return_counts=True)
        assert sorted(nodes.tolist()) == sorted(tree.chance_child[c][tree.chance_child[c] >= 0].tolist()) and len(nodes) == 4
        assert np.abs(counts - 5000).max() < 400
    mixed = np.array([0, start[0], start[1], 0])  # decision nodes are left alone
    dealt = s._deal(mixed.copy())
    assert dealt[1] == start[0] and dealt[2] == start[1] and tree.kind[dealt[0]] == DECISION and tree.kind[dealt[3]] == DECISION


def test_acting_uses_each_players_own_networks_over_the_legal_actions():
    s = _solver("paper")
    tree = s.tree
    _constant(s.Q[0], [9.0, 1.0, 5.0])  # folding looks best to both - where it is legal
    _constant(s.Q[1], [9.0, 5.0, 1.0])
    pi = np.array([[0.7, 0.1, 0.2], [0.2, 0.3, 0.5]])
    for p in (0, 1):
        _constant(s.Pi[p], np.log(pi[p]))
    info = np.tile(np.arange(tree.num_infosets), 3000)
    br = np.arange(len(info)) % 7 < 3  # no pattern shared with the infosets (there are 12)
    a = s._act(info, br, 0.0)
    legal, seat = tree.info_legal[info], tree.info_player[info]
    assert a.shape == info.shape and legal[np.arange(len(a)), a].all()
    fold = legal[:, 0]
    assert set(seat.tolist()) == {0, 1} and fold.any() and not fold.all()
    greedy = np.where(fold, 0, np.where(seat == 0, 2, 1))  # the best LEGAL action of the player's own Q
    assert (a[br] == greedy[br]).all()
    for p in (0, 1):  # average mode: the player's own Pi, renormalised over the legal actions
        for mask in (fold, ~fold):
            rows = ~br & (seat == p) & mask
            want = pi[p] * legal[rows][0]
            freq = np.bincount(a[rows], minlength=3) / rows.sum()
            assert rows.sum() > 3000 and np.abs(freq - want / want.sum()).max() < 0.03, (p, freq)


@pytest.mark.parametrize("shared", [False, True])
def test_exploration_is_uniform_over_the_legal_actions(shared):
    s = _solver("paper", shared_explore=shared)
    tree = s.tree
    for p in (0, 1):
        _constant(s.Q[p], [9.0, 5.0, 1.0])  # greedy: the first legal action
    info = np.tile(np.arange(tree.num_infosets), 20)
    br = np.ones(len(info), dtype=bool)
    legal, seat = tree.info_legal[info], tree.info_player[info]
    first = legal.argmax(1)
    a = s._act(info, br, 1.0)  # eps = 1: every action is exploratory
    assert legal[np.arange(len(a)), a].all() and 0.4 < (a == first).mean() < 0.6  # two legal actions everywhere in Kuhn
    all_greedy, rate = np.zeros((300, 2), dtype=bool), []
    for k in range(300):
        a = s._act(info, br, 0.3)
        rate.append((a != first).mean())
        for p in (0, 1):
            all_greedy[k, p] = (a == first)[seat == p].all()
    assert np.mean(rate) == pytest.approx(0.15, abs=0.02)  # eps / 2: half of the random actions are the greedy one
    if shared:  # the DREAM code: one coin per seat for all of its tables
        assert abs(all_greedy.mean() - 0.7) < 0.08 and (all_greedy[:, 0] != all_greedy[:, 1]).mean() > 0.25
    else:  # the paper: one coin per decision
        assert not all_greedy.any() and np.std(rate) < 0.04


def test_mode_is_drawn_per_hand_and_seat_and_held():
    s = _solver("paper", eta=0.3, envs=64, steps=64)
    assert s.br.shape == (64, 2) and s.br.dtype == bool and 0.15 < s.br.mean() < 0.45  # the first hands draw their modes too
    drawn, mixed, steps = [s.br.copy()], 0, 0
    for _ in range(500):
        before = s.br.copy()
        done = s._play()
        assert np.array_equal(s.br[~done], before[~done])  # held until the hand ends
        drawn.append(s.br[done])
        if done.sum() >= 8:  # tables draw separately: the hands dealt in one step do not share a mode
            steps += 1
            mixed += 0 < s.br[done, 0].sum() < done.sum()
    drawn = np.concatenate(drawn)
    assert len(drawn) > 8000 and mixed > 0.8 * steps > 100
    np.testing.assert_allclose(drawn.mean(0), 0.3, atol=0.02)  # best-response mode with probability eta, either seat
    assert (drawn[:, 0] & drawn[:, 1]).mean() == pytest.approx(0.09, abs=0.015)  # the seats draw independently
    assert (drawn[:, 0] != drawn[:, 1]).mean() == pytest.approx(0.42, abs=0.03)


@pytest.mark.parametrize("preset", ["paper", "dream"])
def test_m_sl_receives_exactly_the_best_response_decisions(preset):
    s = _solver(preset, game="leduc", eta=0.4, eps_start=1.0, eps_const=0.0, envs=64, steps=64, seed=3)  # eps = 1 throughout
    tree, rows = s.tree, np.arange(64)
    calls, want = _spy_actions(s), [[], []]
    for _ in range(80):
        node = s.node.copy()
        info, seat = tree.info[node], tree.player[node]
        br = s.br[rows, seat].copy()
        s._play()
        a, eps = calls[-1]
        assert eps == 1.0
        for p in (0, 1):
            mine = (seat == p) & br
            want[p] += list(zip(info[mine].tolist(), a[mine].tolist()))
    for p in (0, 1):
        m = s.sl_memory[p]
        got = list(zip(m.data["info"][: m.size].tolist(), m.data["action"][: m.size].tolist()))
        assert got == want[p] and m.size == m.seen > 500  # nothing else, nothing missing - exploratory actions included
        assert (tree.info_player[m.data["info"][: m.size]] == p).all()
        taken = np.bincount(m.data["action"][: m.size], minlength=3) / m.size
        assert taken.min() > 0.1  # eps = 1: not the greedy actions only
    assert s.rl_memory[0].seen > 2 * s.sl_memory[0].seen  # M_RL has the average-mode decisions as well


_GAMES = [("paper", "kuhn", 0.4), ("paper", "leduc", 2.6), ("dream", "kuhn", 0.4), ("dream", "leduc", 2.6), ("antes", "leduc", 1.0)]


@pytest.mark.parametrize("preset,game,scale", _GAMES)
def test_every_transition_reaches_m_rl_for_both_seats(preset, game, scale):
    """A step-by-step replay of every table: each player's consecutive decisions are linked by a zero-reward
    transition to its OWN next infoset; the end of a hand completes both players' last decisions with their own
    utility (seat 1: the negative), scaled; nothing leaks into the next hand."""
    kw = dict(preset="paper", reward_scale=1.0) if preset == "antes" else dict(preset=preset)  # both presets: the largest utility / 5
    s = _solver(game=game, eta=0.5, envs=32, steps=32, seed=1, **kw)
    assert s.reward_scale == pytest.approx(scale)
    tree, n = s.tree, s.envs
    calls = _spy_actions(s)
    pending, want, hands = [[None, None] for _ in range(n)], [[], []], 0
    for _ in range(200):
        node = s.node.copy()
        done = s._play()
        a = calls[-1][0]
        for e in range(n):
            h, act = int(node[e]), int(a[e])
            p, i = int(tree.player[h]), int(tree.info[h])
            assert tree.kind[h] == DECISION and tree.legal[h, act]
            if pending[e][p] is not None:
                want[p].append((*pending[e][p], 0.0, i, 0.0))
            pending[e][p] = (i, act)
            child = int(tree.child[h, act])
            if tree.kind[child] == TERMINAL:
                for q in (0, 1):
                    if pending[e][q] is not None:
                        u = float(tree.util[child]) / scale
                        want[q].append((*pending[e][q], round(u if q == 0 else -u, 4), 0, 1.0))
                pending[e], hands = [None, None], hands + 1
                assert done[e] and tree.kind[s.node[e]] == DECISION and tree.player[s.node[e]] == 0  # dealt again
            else:
                assert not done[e] and (s.node[e] == child or (tree.kind[child] == CHANCE and s.node[e] in tree.chance_child[child]))
    assert s.episodes == hands > 300
    for p in (0, 1):
        m = s.rl_memory[p]
        d = {k: v[: m.size] for k, v in m.data.items()}
        got = list(zip(d["info"].tolist(), d["action"].tolist(), np.round(d["reward"].astype(np.float64), 4).tolist(), d["next"].tolist(),
                       d["done"].tolist()))
        assert sorted(got) == sorted(want[p]) and m.size == m.seen == len(want[p])
        assert (tree.info_player[d["info"]] == p).all() and (tree.info_player[d["next"]][d["done"] == 0] == p).all()
        assert not d["reward"][d["done"] == 0].any()
        assert (d["done"] == 0).sum() > 30 or (game, p) == ("kuhn", 1)  # (the second seat of Kuhn never acts twice)
    r0 = s.rl_memory[0].data["reward"][: s.rl_memory[0].size].astype(np.float64)
    r1 = s.rl_memory[1].data["reward"][: s.rl_memory[1].size].astype(np.float64)
    assert np.abs(r0).sum() > 100 / scale and r0.sum() == pytest.approx(-r1.sum(), rel=1e-5)


@pytest.mark.parametrize("eta", [1.0, 0.0])
def test_eta_switches_between_pure_best_response_and_pure_average_play(eta):
    s = _solver("paper", eta=eta)
    for _ in range(20):
        s._play()
    assert s.br.mean() == eta and sum(m.seen for m in s.rl_memory) == 20 * s.envs - int((s.prev_info >= 0).sum())
    if eta:  # every decision is a best-response decision: M_SL has all of them (M_RL lacks only the pending ones)
        assert sum(m.seen for m in s.sl_memory) == 20 * s.envs
    else:
        assert not any(m.seen for m in s.sl_memory) and all(m.seen > 500 for m in s.rl_memory)


# ---- learning ------------------------------------------------------------------------------------------------------
_UPDATE_CASES = [("paper", "leduc", {}), ("dream", "leduc", {}), ("dream", "kuhn", {}),
                 ("paper", "leduc", dict(double_dqn=True, grad_clip=0.3, layers=2, hidden=32)),
                 ("dream", "leduc", dict(double_dqn=False, grad_clip=0.0, lr_q=0.02))]  # (unclipped, the dueling net diverges at 0.1)


@pytest.mark.parametrize("preset,game,kw", _UPDATE_CASES)
def test_updates_equal_torch_autograd(preset, game, kw):
    """One Q update and one Pi update of the solver against the textbook version in torch: the same minibatch through
    the torch modules, the target computed from a separate target network, autograd, clip_grad_norm_, optim.SGD."""
    s = _solver(preset, game=game, seed=2, eta=0.3, target_every=1000, **kw)
    tree = s.tree
    obs, legal = torch.as_tensor(tree.info_obs), torch.as_tensor(tree.info_legal)
    s.iterate(40)

    def sgd_step(module, loss, lr):
        opt = torch.optim.SGD(module.parameters(), lr=lr)
        loss.backward()
        if s.grad_clip:
            torch.nn.utils.clip_grad_norm_(module.parameters(), s.grad_clip)
        opt.step()

    def same(net, module):
        for k, param in module.named_parameters():
            np.testing.assert_allclose(net.w[k], param.detach().numpy(), atol=1e-5, err_msg=k)

    for p in (0, 1):
        if s.arch == "deepcfr_dueling":  # all values negative: the illegal outputs (exactly 0) would win an unmasked max
            s.Q[p].w["v.bias"] -= 5.0
        target = s.Q[p].torch_module()  # the target network: Q as it is now ...
        s._sync_target(p)
        for _ in range(5):  # ... while the online network moves on
            assert s._update_q(p)
        mem = s.rl_memory[p]
        j = np.random.default_rng(5 + p).integers(0, mem.size, s.batch)
        s._batch = lambda m: j
        col = {k: torch.as_tensor(v[j]) for k, v in mem.data.items()}
        nxt, done = col["next"].long(), col["done"]
        module = s.Q[p].torch_module()
        with torch.no_grad():
            tq, oq = target(obs[nxt]), module(obs[nxt])
            pick = torch.where(legal[nxt], oq if s.double_dqn else tq, torch.full_like(tq, -1e30)).argmax(1)
            y = col["reward"] + (1.0 - done) * tq.gather(1, pick[:, None]).squeeze(1)
        pred = module(obs[col["info"].long()]).gather(1, col["action"].long()[:, None]).squeeze(1)
        sgd_step(module, (pred - y).pow(2).mean(), s.lr_q)
        assert s._update_q(p)
        same(s.Q[p], module)
        # what this comparison is able to see: terminal and non-terminal rows, a target network that differs from the
        # online one, and rows where the best action overall is not legal
        live = done == 0
        assert (~live).sum() > 10 and (tq - oq).abs().max() > 1e-3
        if game == "leduc":  # (in Kuhn only the first seat ever acts twice, and rarely)
            assert live.sum() > 10 and ((oq if s.double_dqn else tq).argmax(1) != pick)[live].any()

        mem = s.sl_memory[p]
        j = np.random.default_rng(7 + p).integers(0, mem.size, s.batch)
        info, action = torch.as_tensor(mem.data["info"][j]).long(), torch.as_tensor(mem.data["action"][j]).long()
        module = s.Pi[p].torch_module()
        logits = module(obs[info])
        sgd_step(module, F.cross_entropy(torch.where(legal[info], logits, torch.full_like(logits, -1e20)), action), s.lr_pi)
        assert s._update_pi(p)
        same(s.Pi[p], module)
        assert mem.size >= s.batch and len(set(action.tolist())) > 1


def test_target_refit_is_counted_in_q_updates_per_player():
    s = _solver("paper", target_every=3)
    tree = s.tree

    def values(p):
        return s.Q[p].forward(tree.info_obs, tree.info_legal).copy()

    start = [values(0), values(1)]
    assert all(np.array_equal(s.target_q[p], start[p]) for p in (0, 1))  # Q' <- Q at the start
    for _ in range(6):
        s._play()
    at = {0: start[0]}
    for k in range(1, 8):
        assert s._update_q(0)
        if k % 3 == 0:
            at[k] = values(0)
        assert np.array_equal(s.target_q[0], at[3 * (k // 3)])  # the values at the last refit ...
        assert k % 3 == 0 or not np.array_equal(s.target_q[0], values(0))  # ... which Q has left since
    assert s.q_updates == [7, 0] and np.array_equal(s.target_q[1], start[1])  # each player counts its own updates
    s.iterate(1)  # two more per player: player 0 refits at its 9th, player 1 not yet
    assert s.q_updates == [9, 2] and np.array_equal(s.target_q[0], values(0)) and np.array_equal(s.target_q[1], start[1])
    s.iterate(1)
    assert s.q_updates == [11, 4] and not np.array_equal(s.target_q[0], values(0)) and not np.array_equal(s.target_q[1], start[1])


def test_no_update_before_a_minibatch_is_stored():
    s = _solver("paper", eta=0.5)
    before = [n.state_dict() for n in s.Q + s.Pi]
    s._play()  # two decisions per table: some rows for every memory, fewer than a minibatch (128)
    s._play()
    assert all(0 < m.size < s.batch for m in s.rl_memory + s.sl_memory)
    assert not any([s._update_q(0), s._update_q(1), s._update_pi(0), s._update_pi(1)]) and s.q_updates == [0, 0]
    for net, state in zip(s.Q + s.Pi, before):
        assert all(np.array_equal(net.w[k], state[k]) for k in state)
    for _ in range(40):
        s._play()
    assert all([s._update_q(0), s._update_q(1), s._update_pi(0), s._update_pi(1)]) and s.q_updates == [1, 1]
    assert not any(np.array_equal(net.w[k], state[k]) for net, state in zip(s.Q + s.Pi, before) for k in state)


def _exact_action_values(tree, game, opponent, p):
    """Player p's action values at its infosets against the behaviour table ``opponent``, p best-responding below
    (the fixed point of Q-learning there), and the infosets' reach by chance and the opponent."""
    _, response = best_response(game, TabularPolicy(game, {k: opponent[i] for i, k in enumerate(tree.info_keys)}), p)
    sigma = opponent.copy()
    for i, key in enumerate(tree.info_keys):
        if tree.info_player[i] == p and key in response:
            sigma[i] = response[key]
    v = tree.values(sigma) * (1.0 if p == 0 else -1.0)
    reach = np.zeros(tree.num_nodes)
    reach[0] = 1.0
    num, den = np.zeros(tree.info_legal.shape), np.zeros(tree.num_infosets)
    for i in range(tree.num_nodes):
        if tree.kind[i] == CHANCE:
            k = tree.chance_child[i] >= 0
            reach[tree.chance_child[i, k]] = reach[i] * tree.chance_prob[i, k]
        elif tree.kind[i] == DECISION:
            a, info = tree.legal[i], tree.info[i]
            if tree.player[i] == p:
                reach[tree.child[i, a]] = reach[i]
                num[info, a] += reach[i] * v[tree.child[i, a]]
                den[info] += reach[i]
            else:
                reach[tree.child[i, a]] = reach[i] * opponent[info, a]
    return num / np.maximum(den, 1e-300)[:, None], den


@pytest.mark.parametrize("preset", ["paper", "dream"])
@pytest.mark.parametrize("learner", [0, 1])
def test_q_learning_finds_the_best_response_to_a_fixed_opponent(preset, learner):
    """The reinforcement-learning half on its own, through play, M_RL, targets and updates: against an opponent that
    always plays one known policy, a seat's Q converges to the exact best-response action values (in its reward unit)
    and its greedy action is the best response everywhere.  Either seat: the rewards' sign, the seat's own next
    infoset, the end of the hand."""
    s = _solver(preset, seed=0, eta=0.5, eps_start=0.3, eps_const=0.0)  # the learner: eps-greedy Q or its (untrained) Pi
    tree = s.tree
    logits = np.log([0.2, 0.5, 0.3])
    _constant(s.Pi[1 - learner], logits)
    fixed = np.where(tree.info_legal, np.exp(logits), 0.0)
    fixed /= fixed.sum(1, keepdims=True)
    for _ in range(800):
        s._play()
        s.br[:, 1 - learner] = False  # the opponent never plays its Q
        s._update_q(learner)
        s._update_q(learner)
    exact, reach = _exact_action_values(tree, s.game, fixed, learner)
    mine = np.flatnonzero((tree.info_player == learner) & (reach > 0))
    legal = tree.info_legal[mine]
    q = s.Q[learner].forward(tree.info_obs[mine], legal) * s.reward_scale
    error = np.abs(q - exact[mine])[legal]
    mean, worst = (0.15, 0.6) if preset == "paper" else (0.25, 0.8)  # 6 seeds: up to 0.07, 0.36 (paper) / 0.17, 0.52 (dream) antes
    assert len(mine) == 6 and error.mean() < mean and error.max() < worst, (error.mean(), error.max())
    greedy = np.where(legal, q, -np.inf).argmax(1)
    regret = np.where(legal, exact[mine], -np.inf).max(1) - exact[mine][np.arange(6), greedy]
    assert regret.max() < 0.2 and (regret == 0).sum() >= 5, regret  # greedy = the best response (one near-tie of 0.14 apart)
    assert np.abs(exact[mine][legal]).max() > 1.2 and s.q_updates[1 - learner] == 0


def test_supervised_learning_averages_the_best_response_behaviour():
    """The supervised half on its own: with every hand in best-response mode and a constant Q, Pi learns the eps-greedy
    behaviour that M_SL recorded - the exploratory actions included."""
    s = _solver("paper", seed=0, eta=1.0, eps_start=0.3, eps_const=0.0, lr_pi=0.05)
    tree = s.tree
    for p in (0, 1):
        _constant(s.Q[p], [1.0, 3.0, 2.0])  # greedy: call, legal everywhere
    for _ in range(1500):
        s._play()
        for p in (0, 1):
            s._update_pi(p)
    want = 0.7 * np.eye(3)[1] + 0.3 * tree.info_legal / tree.info_legal.sum(1, keepdims=True)
    table = s.average_policy().table
    got = np.stack([table[k] for k in tree.info_keys])
    assert np.abs(got - want).max() < 0.05, np.abs(got - want).max()
    assert all(m.seen > 20_000 for m in s.sl_memory) and s.q_updates == [0, 0]


def test_iteration_plays_then_updates_q_then_pi():
    s = _solver("dream", steps=192, envs=64, updates=3, eta=0.5)
    log, calls = [], _spy_actions(s)
    for name in ("_play", "_update_q", "_update_pi"):
        def spy(*args, _name=name, _fn=getattr(s, name)):
            log.append((_name, *args))
            return _fn(*args)
        setattr(s, name, spy)
    s.iterate(1)
    assert log == [("_play",)] * 3 + [("_update_q", 0), ("_update_q", 1)] * 3 + [("_update_pi", 0), ("_update_pi", 1)] * 3
    assert (s.iteration, s.nodes_touched) == (1, 192)
    s.iterate(4)
    assert (s.iteration, s.nodes_touched) == (5, 5 * 192) and len(log) == 5 * 15
    assert [eps for _, eps in calls] == [epsilon(t, 0.06, 0.01) for t in range(1, 6) for _ in range(3)]  # this iteration's eps
    assert s.epsilon == epsilon(5, 0.06, 0.01)
    p = _solver("paper")
    calls = _spy_actions(p)
    p.iterate(9)
    assert [eps for _, eps in calls] == [0.06 / np.sqrt(t) for t in range(1, 10)] and p.nodes_touched == 9 * 128
    with pytest.raises(ValueError, match="multiple"):
        _solver("paper", steps=100, envs=64)


def test_presets_set_what_they_claim():
    paper = dict(arch="mlp", hidden=64, layers=1, lr_q=0.1, lr_pi=0.005, grad_clip=0.0, double_dqn=False, reward_scale=None,
                 eps_start=0.06, eps_const=None, shared_explore=False, batch=128, updates=2, steps=128, envs=128,
                 rl_capacity=200_000, sl_capacity=2_000_000, sl_min_prob=0.0, sl_window=False, target_every=300, eta=0.1)
    assert PRESETS["paper"] == paper
    assert {k: v for k, v in PRESETS["dream"].items() if paper[k] != v} == dict(
        arch="deepcfr_dueling", lr_pi=0.01, grad_clip=1.0, double_dqn=True, eps_const=0.01, shared_explore=True)
    for preset in ("paper", "dream"):
        s = _solver(preset, game="leduc")
        want = dict(PRESETS[preset], reward_scale=2.6)  # None: the largest utility / 5 (the DREAM code: stack / 5)
        assert s.config == want and all(getattr(s, k) == v for k, v in want.items())
        assert all(type(n).__name__ == ("MLP" if preset == "paper" else "Dueling") for n in s.Q + s.Pi)
        assert len({id(n) for n in s.Q + s.Pi}) == 4 and not np.array_equal(s.Q[0].theta, s.Q[1].theta)
        if preset == "dream":
            assert [n.policy for n in s.Q + s.Pi] == [False, False, True, True]
        else:
            assert all(n.w["0.weight"].shape == (64, 34) and len(n.w) == 4 for n in s.Q + s.Pi)
        assert all((m.capacity, m.reservoir) == (200_000, False) for m in s.rl_memory)
        assert all((m.capacity, m.reservoir, m.min_prob) == (2_000_000, True, 0.0) for m in s.sl_memory)
        assert s.node.shape == (128,) and (s.prev_info == -1).all()
    s = _solver("paper", lr_q=0.05, sl_window=True, eps_const=0.5, sl_capacity=1000, rl_capacity=500)
    assert (s.lr_q, s.lr_pi, s.config["lr_q"]) == (0.05, 0.005, 0.05) and s.epsilon == pytest.approx(0.06)
    assert not s.sl_memory[0].reservoir and s.sl_memory[1].capacity == 1000 and s.rl_memory[1].capacity == 500
    assert _solver("dream", sl_min_prob=0.25).sl_memory[0].min_prob == 0.25
    with pytest.raises(TypeError, match="learning_rate"):
        _solver("paper", learning_rate=0.1)
    with pytest.raises(ValueError, match="preset"):
        _solver("openspiel")


def test_average_policy_is_the_profile_of_the_pi_networks():
    s = _solver("paper")
    tree = s.tree
    pi = np.array([[0.7, 0.1, 0.2], [0.2, 0.3, 0.5]])
    for p in (0, 1):
        _constant(s.Pi[p], np.log(pi[p]))
        _constant(s.Q[p], [0.0, 0.0, 50.0])  # the best-response networks play no part
    table = s.average_policy().table
    assert len(table) == tree.num_infosets
    for i, key in enumerate(tree.info_keys):
        want = pi[tree.info_player[i]] * tree.info_legal[i]
        np.testing.assert_allclose(table[key], want / want.sum(), atol=1e-6)
        assert table[key].dtype == np.float64 and table[key].sum() == pytest.approx(1.0, abs=1e-12)
    assert s.evaluate() == {"average": exploitability(s.game, s.average_policy())[0]}


@pytest.mark.parametrize("preset,iterations,bound", [("paper", 3000, 0.26), ("dream", 700, 0.2)])
def test_kuhn_converges(preset, iterations, bound):
    """8 / 6 seeds: 0.42-0.51 untrained; paper 0.16-0.20 after 3000 iterations, dream 0.10-0.14 after 700."""
    s = _solver(preset, seed=0)
    start = s.evaluate()["average"]
    assert 0.3 < start < 0.6  # an untrained profile is near uniform play (0.458)
    s.iterate(iterations)
    assert s.evaluate()["average"] < bound and s.episodes > 40 * iterations


# ---- checkpoints and the command line --------------------------------------------------------------------------------
def _same(x, y):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    return np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y


@pytest.mark.parametrize("preset", ["paper", "dream"])
def test_checkpoint_continues_identically_and_shares_nothing(preset, tmp_path):
    kw = dict(target_every=7, eta=0.3, rl_capacity=3000, sl_capacity=400)  # both memories overflow within the test
    a = _solver(preset, seed=0, **kw).iterate(40)
    snap = a.state_dict()
    torch.save(snap, tmp_path / "ck.pt")
    kept = copy.deepcopy(snap)
    b = _solver(preset, seed=0, **kw).load_state_dict(torch.load(tmp_path / "ck.pt", weights_only=False))  # through a file
    c = _solver(preset, seed=0, **kw).load_state_dict(snap)  # in memory
    assert (b.iteration, b.nodes_touched, b.episodes, b.q_updates) == (40, 40 * 128, a.episodes, a.q_updates)
    a.iterate(25)
    assert _same(snap, kept)  # a snapshot does not follow its solver ...
    c.iterate(25)
    assert _same(snap, kept)  # ... nor the solver that loaded it
    assert not _same(a.state_dict(), b.state_dict())
    b.iterate(25)
    final = a.state_dict()
    assert _same(final, b.state_dict()) and _same(final, c.state_dict())  # the same run, bit for bit
    assert a.rl_memory[0].seen > 3000 and a.sl_memory[0].seen > 400 and a.q_updates[0] > 100
    assert a.evaluate() == b.evaluate()
    with pytest.raises(ValueError, match="settings.*eta"):
        _solver(preset, **{**kw, "eta": 0.2}).load_state_dict(snap)
    with pytest.raises(ValueError, match="settings"):
        _solver("dream" if preset == "paper" else "paper", **kw).load_state_dict(snap)


def test_the_seed_determines_the_run():
    a, b, c = (_solver("paper", seed=seed) for seed in (0, 0, 1))
    assert (a.seed, c.seed) == (0, 1)
    assert np.array_equal(a.Q[0].theta, b.Q[0].theta) and not np.array_equal(a.Q[0].theta, c.Q[0].theta)  # the networks
    assert np.array_equal(a.node, b.node) and not np.array_equal(a.node, c.node)  # the deals
    assert np.array_equal(a.br, b.br) and not np.array_equal(a.br, c.br)  # the modes
    for s in (a, b, c):
        s.iterate(5)
    assert _same(a.state_dict(), b.state_dict()) and not _same(a.state_dict(), c.state_dict())


def test_a_checkpoint_of_another_seed_is_refused():
    """Continuing a seed-0 checkpoint as "seed 1" reproduced the seed-0 run under the other label."""
    state = _solver("paper", seed=0).iterate(2).state_dict()
    assert state["seed"] == 0
    with pytest.raises(ValueError, match="settings.*seed = 0 .here: 1."):
        _solver("paper", seed=1).load_state_dict(state)
    assert _solver("paper", seed=0).load_state_dict(state).iteration == 2
    before = {k: v for k, v in state.items() if k != "seed"}  # a checkpoint written before the seed was stored: accepted
    assert _solver("paper", seed=1).load_state_dict(before).iteration == 2


def test_cli_writes_the_curve_and_resumes(tmp_path, capsys):
    from headsup.algos.nfsp import main

    ck, js = tmp_path / "ck.pt", tmp_path / "run.json"
    common = ["--game", "kuhn", "--preset", "paper", "--seed", "3", "--eval-every", "20", "--eta", "0.2"]
    args = common + ["--checkpoint", str(ck), "--checkpoint-minutes", "0", "--json", str(js)]
    main(args + ["--iterations", "40"])
    out = capsys.readouterr().out
    assert "kuhn nfsp paper it 40: exploitability" in out and "nodes 5.12e+03" in out
    run = json.load(open(js))
    assert [c["iteration"] for c in run["curve"]] == [1, 2, 5, 10, 20, 40]  # 1-2-5 up to --eval-every, its multiples, the last
    last = run["curve"][-1]
    assert set(last) == {"iteration", "average", "nodes_touched", "env_steps", "episodes", "epsilon", "seconds"}
    assert last["nodes_touched"] == last["env_steps"] == 40 * 128 and 0.0 < last["average"] < 1.0 and last["episodes"] > 1000
    assert last["epsilon"] == pytest.approx(0.06 / 40**0.5)
    assert (run["game"], run["algo"], run["preset"]) == ("kuhn", "nfsp", "paper")
    assert run["config"]["eta"] == 0.2 and run["config"]["lr_pi"] == 0.005 and run["args"]["seed"] == 3
    assert ck.exists() and not (tmp_path / "ck.pt.tmp").exists()
    main(args + ["--iterations", "70"])  # resumes at 40
    assert "resumed from" in capsys.readouterr().out
    curve = json.load(open(js))["curve"]
    assert [c["iteration"] for c in curve] == [1, 2, 5, 10, 20, 40, 60, 70]
    straight = tmp_path / "straight.json"
    main(common + ["--json", str(straight), "--iterations", "70"])  # no checkpoint: one uninterrupted run
    assert [c["average"] for c in json.load(open(straight))["curve"]] == [c["average"] for c in curve]
    with pytest.raises(ValueError, match="settings"):
        main(["--game", "kuhn", "--preset", "paper", "--seed", "3", "--eta", "0.5", "--checkpoint", str(ck), "--iterations", "80"])
    with pytest.raises(SystemExit):
        main(["--game", "kuhn", "--device", "cuda", "--iterations", "1"])
    capsys.readouterr()


def test_cli_overrides_reach_the_solver(tmp_path, capsys):
    from headsup.algos.nfsp import main

    js = tmp_path / "run.json"
    main(["--game", "kuhn", "--preset", "dream", "--iterations", "3", "--json", str(js), "--no-double-dqn", "--eps-const", "none",
          "--reward-scale", "2", "--no-shared-explore", "--arch", "mlp", "--hidden", "16", "--layers", "2", "--lr-q", ".05",
          "--lr-pi", ".02", "--grad-clip", "0", "--batch", "32", "--updates", "1", "--steps", "64", "--envs", "32", "--rl-capacity", "1000",
          "--sl-capacity", "500", "--sl-min-prob", "0.25", "--sl-window", "--target-every", "10", "--eta", "0.5", "--eps-start", "0.1"])
    run = json.load(open(js))
    assert run["config"] == dict(arch="mlp", hidden=16, layers=2, lr_q=0.05, lr_pi=0.02, grad_clip=0.0, double_dqn=False, reward_scale=2.0,
                                 eps_start=0.1, eps_const=None, shared_explore=False, batch=32, updates=1, steps=64, envs=32,
                                 rl_capacity=1000, sl_capacity=500, sl_min_prob=0.25, sl_window=True, target_every=10, eta=0.5)
    assert run["curve"][-1]["nodes_touched"] == 3 * 64 and run["curve"][-1]["epsilon"] == pytest.approx(0.1 / 3**0.5)
    main(["--game", "kuhn", "--preset", "paper", "--iterations", "2", "--json", str(js), "--reward-scale", "auto", "--eps-const", "0.01"])
    config = json.load(open(js))["config"]
    assert config["reward_scale"] == pytest.approx(0.4) and config["eps_const"] == 0.01 and config["lr_pi"] == 0.005
    capsys.readouterr()
