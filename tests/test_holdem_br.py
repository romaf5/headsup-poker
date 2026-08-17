"""Hold'em best-response exploitability estimator (headsup.algos.holdem_br)."""

from math import comb

import numpy as np
import pytest
import torch

from headsup import native
from headsup.algos.holdem_br import HoldemBestResponse, build_street_tree, chance_factor, main
from headsup.engine import HeadsUpPoker
from headsup.game import DEFAULT_GAME, FHP
from headsup.lbr import substitute_hands, valid_combos
from headsup.model import BaseModel
from headsup.players import TorchPolicyPlayer, make_player

BOARD = [4, 5, 6, 7, 8]


def _random_player(game, seed=0):
    torch.manual_seed(seed)
    m = BaseModel(game=game)
    with torch.no_grad():
        torch.nn.init.normal_(m.action_head.weight, std=0.5)
    return TorchPolicyPlayer(m, device="cpu")


def test_chance_factor_is_the_pair_compatibility_probability():
    assert chance_factor(0, 3) == pytest.approx(comb(48, 3) / comb(52, 3))
    assert chance_factor(0, 5) == pytest.approx(comb(48, 5) / comb(52, 5))
    assert chance_factor(3, 5) == pytest.approx(comb(48, 5) / comb(52, 5) / (comb(48, 3) / comb(52, 3)))
    assert chance_factor(5, 5) == 1.0


def test_street_trees_have_street_end_leaves_then_showdowns():
    e = HeadsUpPoker(rng=np.random.default_rng(0), game=FHP)
    e.reset(list(range(9)))
    pre = build_street_tree(e)
    kinds = [nd.kind for nd in pre]
    assert kinds.count(3) > 0 and kinds.count(1) > 0 and kinds.count(2) == 0  # limit: no all-in showdown pre-flop
    for nd in pre:
        if nd.kind == 3:
            assert int(nd.engine.stage) == 1 and not nd.engine.done and nd.engine.current == 1  # BB first post-flop
    e.step(1)
    e.step(1)  # flop = FHP's last street
    flop = build_street_tree(e)
    kinds = [nd.kind for nd in flop]
    assert kinds.count(2) > 0 and kinds.count(1) > 0 and kinds.count(3) == 0


@pytest.mark.skipif(not native.available(), reason="C++ extension not built")
def test_river_values_match_the_exact_search_evaluator():
    """On a river subgame (no chance left) the estimator with the single real board must reproduce
    the exact evaluator of headsup.search on the C++ solver's tree, node by node."""
    from headsup.search import exploitability

    cpp = native.module()
    player = _random_player(DEFAULT_GAME)
    e = HeadsUpPoker(rng=np.random.default_rng(0), game=DEFAULT_GAME)
    e.reset(list(range(9)))
    for a in [1, 1, 1, 1, 1, 1]:
        e.step(a)
    ce = cpp.Engine()
    ce.reset(list(range(9)))
    for a in [1, 1, 1, 1, 1, 1]:
        ce.step(a)
    sv = cpp.SubgameSolver()
    sv.build(ce)
    tree = sv.tree()
    # strategies at every node of the C++ tree = the same player queried on the replayed engine
    strat, stack = {}, [(0, e.clone())]
    while stack:
        i, eng = stack.pop()
        nd = tree[i]
        if nd["kind"] != 0:
            continue
        strat[i] = np.asarray(player.probs(substitute_hands(eng.observation(nd["player"]))), dtype=np.float64)
        for a in range(DEFAULT_GAME.num_actions):
            if nd["legal"][a]:
                c = eng.clone()
                c.step(a)
                stack.append((nd["child"][a], c))
    r = valid_combos(BOARD).astype(np.float64)
    ex_ref, br_ref, v_ref = exploitability(tree, strat, [r, r], BOARD)
    res = HoldemBestResponse(player, DEFAULT_GAME, boards=1).evaluate_from(e, [r, r], [tuple(BOARD)])
    assert res["exploitability_chips"] == pytest.approx(ex_ref, rel=1e-9)
    assert res["br_values"] == pytest.approx(list(br_ref), rel=1e-9)
    assert res["values"] == pytest.approx(list(v_ref), rel=1e-9)


def test_fhp_estimate_is_zero_sum_and_bounds_the_profile():
    player = _random_player(FHP, seed=1)
    res = HoldemBestResponse(player, FHP, boards=3, seed=0).run()
    v0, v1 = res["values"]
    assert v0 == pytest.approx(-v1, abs=1e-9)  # exact zero sum for any sample of boards
    assert res["br_values"][0] >= v0 - 1e-9 and res["br_values"][1] >= v1 - 1e-9
    assert res["exploitability_mbb"] > 0 and res["exploitability_mbb"] == pytest.approx(1000 * res["exploitability_chips"] / FHP.big_blind)
    # always-call: symmetric game value 0, both best responses equal by symmetry of a fixed strategy? no -
    # positions differ; but zero-sum and a positive gain hold
    call = HoldemBestResponse(make_player("call", game=FHP), FHP, boards=2, seed=0).run()
    assert call["values"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert min(call["br_values"]) > 0


def test_cli_runs(capsys):
    main(["--policy", "call", "--game", "fhp", "--boards", "1", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "exploitability" in out and "mbb/g" in out
