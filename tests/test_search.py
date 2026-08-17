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
    sv = cpp.RiverSolver()
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
