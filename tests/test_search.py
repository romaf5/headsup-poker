"""Real-time subgame search: solver correctness (exact best responses on river subgames) and the player."""

import numpy as np
import pytest
import torch

from headsup import native
from headsup.cards import hand_strength
from headsup.lbr import COMBOS, NUM_COMBOS, valid_combos
from headsup.search import _opponent_mass, _showdown_values, exploitability, parse_search_spec

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")

BOARD = [4, 5, 6, 7, 8]


def _river_root():
    cpp = native.module()
    e = cpp.Engine()
    e.reset(list(range(9)))  # seat 0: cards 0, 1; seat 1: 2, 3; board 4..8
    for a in [1, 1, 1, 1, 1, 1]:  # checked down to the river, BB (seat 1) to act
        e.step(a)
    return e


def test_vector_payoff_helpers_match_brute_force():
    rng = np.random.default_rng(0)
    ok = valid_combos(BOARD)
    reach = np.where(ok, rng.random(NUM_COMBOS), 0.0)
    strength = np.full(NUM_COMBOS, np.inf)
    for h in np.flatnonzero(ok):
        strength[h] = hand_strength([int(COMBOS[h, 0]), int(COMBOS[h, 1])], BOARD)
    mass = _opponent_mass(reach)
    sd = _showdown_values(reach, strength)
    compat = ~((COMBOS[:, None, 0] == COMBOS[None, :, 0]) | (COMBOS[:, None, 0] == COMBOS[None, :, 1])
               | (COMBOS[:, None, 1] == COMBOS[None, :, 0]) | (COMBOS[:, None, 1] == COMBOS[None, :, 1]))
    for h in rng.choice(np.flatnonzero(ok), 40, replace=False):
        others = np.flatnonzero(compat[h] & ok)
        assert mass[h] == pytest.approx(reach[others].sum())
        wins = reach[others][strength[others] > strength[h]].sum()
        loses = reach[others][strength[others] < strength[h]].sum()
        assert sd[h] == pytest.approx(wins - loses)


def test_solver_converges_on_a_river_subgame():
    cpp = native.module()
    e = _river_root()
    r = valid_combos(BOARD).astype(np.float32)
    sv = cpp.SubgameSolver()
    sv.build(e)
    assert sv.num_nodes == 55 and sv.num_leaves == 0
    sv.set_ranges(r, r.copy())
    tree = sv.tree()
    dec = [i for i, nd in enumerate(tree) if nd["kind"] == 0]
    uniform = {}
    for i in dec:
        row = np.array(tree[i]["legal"], dtype=np.float64)
        uniform[i] = np.repeat((row / row.sum())[None], NUM_COMBOS, 0)
    ex_uniform = exploitability(tree, uniform, [r, r], BOARD)[0]
    sv.run(200_000, 1)
    strat = {i: sv.node_strategy(i).astype(np.float64) for i in dec}
    ex, br, v = exploitability(tree, strat, [r, r], BOARD)
    assert v[0] == pytest.approx(-v[1])  # zero sum
    assert br[0] >= v[0] - 1e-9 and br[1] >= v[1] - 1e-9  # best responses cannot do worse than the profile
    assert ex < 0.6 * ex_uniform, (ex, ex_uniform)  # ~1.2 vs 2.7 chips per hand pair
    for i in dec:
        np.testing.assert_allclose(strat[i].sum(1), 1.0, atol=1e-5)
        assert np.all(strat[i][:, ~np.array(tree[i]["legal"])] == 0)


def test_focused_sampling_solves_the_real_hand():
    """With the hero's real hand dealt on half of its traversals, that hand's regret against the
    solved opponent is ~0 while the opponent's solve stays a range-wide one."""
    cpp = native.module()
    e = _river_root()
    r = valid_combos(BOARD).astype(np.float64)
    strength = np.full(NUM_COMBOS, np.inf)
    for h in np.flatnonzero(r > 0):
        strength[h] = hand_strength([int(COMBOS[h, 0]), int(COMBOS[h, 1])], BOARD)
    hh = cpp.combo_index(2, 3)  # the acting player's (seat 1) real hand

    def real_hand_gap(strat, tree):
        def values(node, reach_q, br):
            nd = tree[node]
            if nd["kind"] == 1:
                return (1.0 if nd["folder"] != 1 else -1.0) * nd["stake"] * _opponent_mass(reach_q)
            if nd["kind"] == 2:
                return nd["stake"] * _showdown_values(reach_q, strength)
            acts = [a for a in range(4) if nd["legal"][a]]
            if nd["player"] == 1:
                cv = np.stack([values(nd["child"][a], reach_q, br) for a in acts])
                return cv.max(0) if br else (strat[node][:, acts].T * cv).sum(0)
            return sum(values(nd["child"][a], reach_q * strat[node][:, a], br) for a in acts)

        m = _opponent_mass(r)[hh]
        return (values(0, r, True)[hh] - values(0, r, False)[hh]) / m

    gaps = {}
    for focus in (0.0, 0.5):
        sv = cpp.SubgameSolver()
        sv.build(e)
        sv.set_ranges(r.astype(np.float32), r.astype(np.float32))
        sv.run(50_000, 3, 1, hh, focus)
        tree = sv.tree()
        strat = {i: sv.node_strategy(i).astype(np.float64) for i, nd in enumerate(tree) if nd["kind"] == 0}
        gaps[focus] = real_hand_gap(strat, tree)
    assert gaps[0.5] < 0.05 and gaps[0.5] < 0.1 * gaps[0.0], gaps


def test_warm_start_from_the_previous_solve_helps():
    cpp = native.module()
    root = _river_root()
    r = valid_combos(BOARD).astype(np.float32)
    first = cpp.SubgameSolver()
    first.build(root)
    first.set_ranges(r, r.copy())
    first.run(100_000, 1)
    # BB checks, SB bets: the new root is a node of the previous tree
    e2 = cpp.Engine()
    e2.reset(list(range(9)))
    for a in [1, 1, 1, 1, 1, 1, 1, 2]:
        e2.step(a)
    old_node = first.child(first.child(0, 1), 2)
    assert old_node > 0
    r_bb = (r * first.node_strategy(0)[:, 1]).astype(np.float32)
    r_sb = (r * first.node_strategy(first.child(0, 1))[:, 2]).astype(np.float32)
    ex = {}
    for warm in (0, 5000):
        sv = cpp.SubgameSolver()
        sv.build(e2)
        sv.set_ranges(r_sb, r_bb)
        if warm:
            sv.warm_start(first, old_node, warm)
        sv.run(5000, 2)
        tree = sv.tree()
        strat = {i: sv.node_strategy(i).astype(np.float64) for i, nd in enumerate(tree) if nd["kind"] == 0}
        ex[warm] = exploitability(tree, strat, [r_sb, r_bb], BOARD)[0]
    assert ex[5000] < 0.5 * ex[0], ex


@pytest.mark.parametrize("variant", ["lcfr", "dcfr", "cfr+", "pcfr+"])
def test_vector_river_solver_is_near_exact(variant):
    cpp = native.module()
    e = _river_root()
    r = valid_combos(BOARD).astype(np.float32)
    sv = cpp.VectorSolver()
    sv.build(e)
    sv.set_ranges(r, r.copy())
    sv.set_variant(variant)
    sv.run(200)
    tree = sv.tree()
    dec = [i for i, nd in enumerate(tree) if nd["kind"] == 0]
    strat = {i: sv.node_strategy(i).astype(np.float64) for i in dec}
    ex, br, v = exploitability(tree, strat, [r, r], BOARD)
    assert ex < 0.02, ex  # sampled MCCFR needs ~800k iterations for 0.27; uniform is 2.7
    assert v[0] == pytest.approx(-v[1], abs=1e-6)
    for i in dec:
        np.testing.assert_allclose(strat[i].sum(1), 1.0, atol=1e-5)


def test_pre_river_subgame_needs_and_uses_continuations():
    from headsup.model import BaseModel

    cpp = native.module()
    e = cpp.Engine()
    e.reset(list(range(9)))
    e.step(1)
    e.step(1)  # flop, BB to act
    sv = cpp.SubgameSolver()
    sv.build(e)
    assert sv.num_leaves == 5 and 40 < sv.num_nodes < 70
    r = valid_combos([4, 5, 6]).astype(np.float32)
    sv.set_ranges(r, r.copy())
    with pytest.raises(RuntimeError):
        sv.run(10, 0)
    torch.manual_seed(0)
    nets = [native.make_model(BaseModel().numpy_weights()) for _ in range(2)]
    sv.set_continuations([nets[0]], [nets[1]], [True], [1.0])
    sv.run(300, 0, 1, cpp.combo_index(2, 3), 0.5)
    st = sv.root_strategy()
    np.testing.assert_allclose(st.sum(1), 1.0, atol=1e-5)
    assert np.all(st[:, 0] == 0)  # nothing to call for the BB: never fold
    # Pluribus's leaf choice: both players pick among 4 biased continuations at every leaf
    sv2 = cpp.SubgameSolver()
    sv2.build(e)
    sv2.set_ranges(r, r.copy())
    sv2.set_continuations([nets[0]], [nets[1]], [True], [1.0])
    sv2.set_leaf_choices(4)
    sv2.run(300, 0, 1, cpp.combo_index(2, 3), 0.5)
    st2 = sv2.root_strategy()
    np.testing.assert_allclose(st2.sum(1), 1.0, atol=1e-5)
    with pytest.raises(RuntimeError):
        sv2.set_leaf_choices(5)
    assert parse_search_spec("cfr:x.pth@leaf4@it100") == ("cfr:x.pth", {"leaf_choices": 4, "iterations": 100})
    assert parse_search_spec("sdcfr:x/it.pt@k64@it100") == ("sdcfr:x/it.pt@k64", {"iterations": 100})  # bank thinning


def test_search_spec_and_player_bookkeeping(tmp_path):
    from headsup.env import make_vec_env, play_hands
    from headsup.model import BaseModel
    from headsup.players import make_player

    assert parse_search_spec("cfr:x.pth@it500@focus0.3@contpolicy") == ("cfr:x.pth", {"iterations": 500, "focus": 0.3, "continuation": "policy"})
    assert parse_search_spec("sdcfr:it.pt@g2@it100@thin4") == ("sdcfr:it.pt@g2", {"iterations": 100, "thin": 4})
    assert parse_search_spec("cfr:x.pth@rit50@rvcfr+") == ("cfr:x.pth", {"river_iterations": 50, "river_variant": "cfr+"})
    torch.manual_seed(0)
    m = BaseModel()
    with torch.no_grad():
        torch.nn.init.normal_(m.action_head.weight, std=0.3)
    m.save(tmp_path / "bp.pth")
    p = make_player(f"search:cfr:{tmp_path / 'bp.pth'}@it300@rit20", device="cpu", seed=0)
    assert p.wants_ids and p.game.num_actions == 4 and p.continuation == "policy"
    env = make_vec_env(4, "call", seed=0, game=p.game)
    r = play_hands(env, p, 8)
    assert len(r) == 8
    for st in p.state.values():
        assert st["villain"].sum() == pytest.approx(1.0) and st["hero"].sum() == pytest.approx(1.0)
        assert np.all(st["villain"][valid_combos(list(st["cards"])) == 0] == 0)  # the villain cannot hold our cards


def test_pluribus_mode_player(tmp_path):
    """Pluribus mode: the blueprint plays preflop, later streets re-solve the remaining game from
    the round start (frozen own actions), ranges are updated at round boundaries."""
    from headsup.env import make_vec_env, play_hands
    from headsup.model import BaseModel
    from headsup.players import TorchPolicyPlayer, make_player

    torch.manual_seed(0)
    m = BaseModel()
    with torch.no_grad():
        torch.nn.init.normal_(m.action_head.weight, std=0.3)
    m.save(tmp_path / "bp.pth")
    spec = f"search:cfr:{tmp_path / 'bp.pth'}@pluribus@it40@b50@th2"
    assert parse_search_spec(spec)[1] == {"mode": "pluribus", "iterations": 40, "buckets": 50, "threads": 2}
    p = make_player(spec, device="cpu", seed=0)
    assert p.mode == "pluribus" and p.iterations == 40 and p.play == "final" and p.preflop == "blueprint"
    # preflop: exactly the blueprint's distribution
    from headsup.engine import HeadsUpPoker

    e = HeadsUpPoker(rng=np.random.default_rng(1))
    e.reset()
    obs = e.observation(0)[None]
    bp = TorchPolicyPlayer(m, device="cpu")
    np.testing.assert_allclose(p.probs(obs, np.array([7])), bp.probs(obs), atol=1e-6)
    env = make_vec_env(4, "call", seed=0, game=p.game)
    r = play_hands(env, p, 16)
    assert len(r) == 16 and p.solves > 0
    for st in p.state.values():
        assert st["villain"].sum() == pytest.approx(1.0) and st["hero"].sum() == pytest.approx(1.0)
        assert np.all(st["villain"][valid_combos(list(st["cards"])) == 0] == 0)
        if st["solver"] is not None:  # a full-game vector solve rooted at the round it was made in (snapshot)
            assert st["solver"].root_round == st["solver_round"] >= 1 and st["solver"].node_player(0) >= 0


def test_search_options_are_found_anywhere_and_unknown_options_raise():
    """Search options left of a blueprint option stayed in the blueprint spec, whose parsers ignored unknown
    options: 'sdcfr:x@pluribus@exact' silently ran depth mode."""
    from headsup.blueprint import parse_tab_spec
    from headsup.players import parse_sdcfr_spec

    assert parse_search_spec("sdcfr:x/it.pt@pluribus@exact") == ("sdcfr:x/it.pt@exact", {"mode": "pluribus"})
    assert parse_search_spec("sdcfr:x/it.pt@it100@k64") == ("sdcfr:x/it.pt@k64", {"iterations": 100})
    assert parse_search_spec("tab:bp.pt@pluribus@current@th2") == ("tab:bp.pt@current", {"mode": "pluribus", "threads": 2})
    assert parse_search_spec("tab:bp.pt@it50@it20")[1] == {"iterations": 20}  # a repeated option: the rightmost wins
    assert parse_sdcfr_spec("x/it.pt@exact@g2@t100@k64") == ("x/it.pt", "exact", 2.0, 100, 64)
    for bad in ("x/it.pt@pluribus", "x/it.pt@t10x", "x/it.pt@exat"):
        with pytest.raises(ValueError, match="option"):
            parse_sdcfr_spec(bad)
    assert parse_tab_spec("bp.pt@current") == ("bp.pt", True)
    with pytest.raises(ValueError, match="option"):
        parse_tab_spec("bp.pt@curent")


def _random_bank(path, T=3, tag=False):
    from headsup.model import BaseModel, normalize_config
    from headsup.sdcfr import IterateBank

    cfg = normalize_config({"features": "history"})
    torch.manual_seed(0)
    dicts = [[], []]
    for seat in (0, 1):
        for t in range(T):
            m = BaseModel(config=cfg)
            with torch.no_grad():
                if tag:
                    m.action_head.bias.fill_(float(t + 1))  # the bias tells which iteration's net it is (the bank holds 1..T)
                else:
                    torch.nn.init.normal_(m.action_head.weight, std=1.5)
                    torch.nn.init.normal_(m.action_head.bias, std=1.5)
            dicts[seat].append({k: v.clone() for k, v in m.state_dict().items()})
    IterateBank.from_state_dicts(dicts, "cpu", cfg).save(path)
    return IterateBank.load(path, "cpu")


def test_iterate_blueprint_and_its_continuation_use_the_same_iterate(tmp_path, monkeypatch):
    """`search:iterate:<bank>@tN`: the opponent model played bank index N, the continuation index N - 1."""
    import headsup.search as search
    from headsup.players import make_player

    path = tmp_path / "bank.pt"
    _random_bank(path, T=6, tag=True)
    seen = []
    make_model = native.make_model

    def spy(weights):
        seen.extend(float(np.asarray(v).ravel()[0]) for k, v in weights.items() if k.endswith("action_head.bias"))
        return make_model(weights)

    monkeypatch.setattr(native, "make_model", spy)
    for n in (2, 4):
        played = float(make_player(f"iterate:{path}@t{n}", device="cpu").nets[0].action_head.bias[0].item())
        seen.clear()
        search._continuations(f"iterate:{path}@t{n}", "cpu", "iterate", 8)
        assert played == n and seen and set(seen) == {float(n)}
        seen.clear()  # an SD-CFR blueprint after n iterations: its last network for the "iterate" continuation
        search._continuations(f"sdcfr:{path}@t{n}", "cpu", "iterate", 8)
        assert set(seen) == {float(n)}
    seen.clear()  # "bank" continuation: representatives of the iterations <= 4, weighted by iteration
    _, _, _, w = search._continuations(f"sdcfr:{path}@t4", "cpu", "bank", 8)
    assert sorted(set(seen)) == [1.0, 2.0, 3.0, 4.0] and w == [1.0, 2.0, 3.0, 4.0]


def test_pluribus_mode_plays_the_sdcfr_average_at_a_second_preflop_decision(tmp_path):
    """Pluribus mode with an SD-CFR blueprint: at the hero's second pre-flop decision the blueprint must weight the
    iterates by their probability of the hero's first action (the reach was never updated: the plain mixture)."""
    from headsup.engine import HeadsUpPoker
    from headsup.players import make_player

    path = tmp_path / "bank.pt"
    bank = _random_bank(path, T=3)
    p = make_player(f"search:sdcfr:{path}@pluribus@it20@b20@th1", device="cpu", seed=0)
    w = bank.weights.numpy()
    for deal_seed in range(200):  # a deal where the iterates raise with clearly different probabilities
        e = HeadsUpPoker(rng=np.random.default_rng(deal_seed), game=p.game)
        e.reset()
        obs1 = e.observation(0)
        sig1 = bank.strategies(0, obs1[None])[:, 0].numpy()
        if (sig1[:, 2] > 0.05).sum() >= 2 and np.ptp(sig1[:, 2]) > 0.3:
            break
    ids = np.array([5])
    np.testing.assert_allclose(p.probs(obs1[None], ids)[0], (w[:, None] * sig1).sum(0) / w.sum(), atol=1e-5)
    e.step(2)  # the hero (small blind) raises
    e.step(2)  # the big blind re-raises
    obs2 = e.observation(0)
    sig2 = bank.strategies(0, obs2[None])[:, 0].numpy()
    reach = w * sig1[:, 2]
    exact = (reach[:, None] * sig2).sum(0) / reach.sum()
    assert np.abs(exact - (w[:, None] * sig2).sum(0) / w.sum()).max() > 0.05  # the reach matters in this deal
    np.testing.assert_allclose(p.probs(obs2[None], ids)[0], exact, atol=1e-5)
    np.testing.assert_allclose(p.probs(obs2[None], ids)[0], exact, atol=1e-5)  # asking again changes nothing


def _random_policy(path, std=0.6):
    from headsup.model import BaseModel
    from headsup.players import TorchPolicyPlayer

    torch.manual_seed(0)
    m = BaseModel()
    with torch.no_grad():
        torch.nn.init.normal_(m.action_head.weight, std=std)
    m.save(path)
    return TorchPolicyPlayer(m, device="cpu")


def test_preflop_search_in_pluribus_mode_gets_the_ranges_of_its_root(tmp_path):
    """`@pluribus@pfsearch`: the pre-flop solve is rooted at the current decision, so the villain's range must
    already contain its actions of this round (it got the round-start prior)."""
    from headsup.engine import HeadsUpPoker
    from headsup.lbr import substitute_hands
    from headsup.players import make_player

    bp = _random_policy(tmp_path / "bp.pth")
    p = make_player(f"search:cfr:{tmp_path / 'bp.pth'}@pluribus@pfsearch@it300@b30@th1", device="cpu", seed=0)
    e = HeadsUpPoker(rng=np.random.default_rng(1), game=p.game)
    e.reset()
    obs_sb = e.observation(0)
    e.step(2)  # the small blind raises; the hero is the big blind
    p.probs(e.observation(1)[None], np.array([0]))
    st = p.state[0]
    assert st["solver_actions"] == [2]  # the solve is rooted after the raise
    post = valid_combos(e.hands[1]) * np.asarray(bp.probs(substitute_hands(obs_sb)), dtype=float)[:, 2]
    np.testing.assert_allclose(st["villain"], post / post.sum(), atol=1e-9)


def test_a_new_hand_with_the_same_hole_cards_starts_from_fresh_ranges(tmp_path):
    """A new hand was recognised by fewer actions or other hole cards only: the same cards in the same seat again
    (duplicate tables, random seats) kept the previous hand's ranges."""
    from headsup.engine import HeadsUpPoker
    from headsup.lbr import substitute_hands
    from headsup.players import make_player

    bp = _random_policy(tmp_path / "bp.pth")
    p = make_player(f"search:cfr:{tmp_path / 'bp.pth'}@it200@rit20", device="cpu", seed=0)
    e = HeadsUpPoker(game=p.game)
    e.reset([0, 1, 20, 33, 5, 6, 7, 8, 9])  # the hero (big blind) holds cards 20, 33
    e.step(2)  # hand 1: the small blind raises, the hero decides once
    p.probs(e.observation(1)[None], np.array([0]))
    e.reset([10, 11, 20, 33, 40, 41, 42, 43, 44])  # the next hand at this table: the same hole cards
    obs_sb = e.observation(0)
    e.step(1)  # hand 2: the small blind limps
    p.probs(e.observation(1)[None], np.array([0]))
    post = valid_combos([20, 33]) * np.asarray(bp.probs(substitute_hands(obs_sb)), dtype=float)[:, 1]
    np.testing.assert_allclose(p.state[0]["villain"], post / post.sum(), atol=1e-9)
