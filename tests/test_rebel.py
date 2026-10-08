"""ReBeL (Brown et al. 2020) on Kuhn / Leduc: public trees and ranges, Linear CFR-D with leaf values, the self-play
data, the value network, test-time play and its exact evaluation."""

import io
import itertools
import json

import numpy as np
import pytest
import torch

from headsup.algos.best_response import TabularPolicy, expected_value, exploitability
from headsup.algos.rebel import (
    AFTER_BOARD,
    BEFORE_BOARD,
    CFRD,
    DECISION,
    EPS,
    FOLD,
    LEAF,
    ROOT,
    SHOWDOWN,
    Playthrough,
    ReBeL,
    Replay,
    bayes_update,
    board_distribution,
    chance_average,
    deal_board,
    huber,
    main,
    normalise,
    sample_board,
    sample_leaf,
    stop_weights,
    value_net,
)
from headsup.algos.tabular import CFR
from headsup.games import make_game

H = 6  # Leduc: a hand is one of six cards


def _leduc(**kw):
    kw = {"iters": 8, "games": 32, "steps": 2, "batch": 16, "capacity": 4096, "eval_samples": 8, "probe_iters": 8, **kw}
    return ReBeL(make_game("leduc"), **kw)


def _strategy(tree, rng, batch=1, hands=H):
    """Random behaviour strategies (batch, D, A, H) over the legal actions."""
    s = rng.random((batch, tree.num_decisions, tree.num_actions, hands)) * tree.legal_f
    return s / s.sum(2, keepdims=True)


def _mix(game, policies, weights):
    """The policy that plays one of ``policies`` (drawn with ``weights``) for the whole game, by brute force over the
    game tree: per infoset the behaviour strategies averaged with weight x the acting player's own reach."""
    weights, num = np.asarray(weights, dtype=np.float64), {}

    def walk(state, reach):
        if state.is_terminal():
            return
        if state.is_chance():
            for a, _ in state.chance_outcomes():
                walk(state.child(a), reach)
            return
        p = state.current_player
        probs = np.stack([np.asarray(pol(state), dtype=np.float64) for pol in policies])
        num.setdefault(state.info_key(p), ((weights * reach[:, p])[:, None] * probs).sum(0))
        for a in state.legal_actions():
            nxt = reach.copy()
            nxt[:, p] *= probs[:, a]
            walk(state.child(a), nxt)

    walk(game.new_initial_state(), np.ones((len(policies), 2)))
    return {k: v / v.sum() for k, v in num.items() if v.sum() > 0}


# ----------------------------------------------------------------------------- trees and ranges
def test_round_trees_match_the_games():
    s = _leduc()
    t0, t1 = s.tree0, s.tree1
    assert (t0.num_nodes, t0.num_decisions, len(t0.leaves), len(t0.folds), len(t0.shows)) == (15, 6, 5, 4, 0)
    assert (t1.num_nodes, t1.num_decisions, len(t1.leaves), len(t1.folds), len(t1.shows)) == (15, 6, 0, 4, 5)
    assert [t0.hist[n][0] for n in t0.leaves] == [(1, 1), (1, 2, 1), (1, 2, 2, 1), (2, 1), (2, 2, 1)]  # cc crc crrc rc rrc
    assert t0.pot[t0.leaves].tolist() == [2, 6, 10, 6, 10] and t0.pot[0] == 2  # the first round's bets are 2
    assert t0.amount[t0.folds].tolist() == [1, 3, 1, 3]  # the folder loses what it has put in
    for leaf, tree in enumerate(s.trees1):  # the last round: bets of 4 on top of the leaf's pot
        base = t0.pot[t0.leaves][leaf] / 2
        assert tree.amount[tree.shows].tolist() == [base, base + 4, base + 8, base + 4, base + 8]
        assert tree.amount[tree.folds].tolist() == [base, base + 4, base, base + 4]
        assert all(h[0] == t0.hist[t0.leaves[leaf]][0] for h in tree.hist)  # its histories continue the leaf's
    assert s.scale == 13.0 and s.in_dim == 3 + H + 2 * H
    for t in (t0, t1):
        d = t.kind == DECISION
        assert (t.child[t.legal] > np.nonzero(t.legal)[0]).all()  # children come after their parents
        assert (t.player[d] >= 0).all() and (t.player[~d] == -1).all() and (t.legal[~d] == False).all()  # noqa: E712
        assert set(np.unique(t.kind[~d])) <= {FOLD, SHOWDOWN, LEAF} and t.index[t.hist[7]] == 7
        assert t.player[t.child[0, 1]] == 1 and t.player[0] == 0  # player 0 opens every round
        assert not t.legal[0, 0] and t.legal[t.child[0, 2], 0]  # a fold only against a bet
    k = ReBeL(make_game("kuhn"), iters=8)
    assert (k.tree0.num_decisions, len(k.tree0.leaves), len(k.tree0.folds), len(k.tree0.shows)) == (4, 0, 2, 3)
    assert k.trees1 == [] and k.scale == 2.0 and k.tree0.amount[k.tree0.shows].tolist() == [1, 2, 2]


def test_reach_is_the_product_of_own_action_probabilities():
    t = _leduc().tree0
    sigma = _strategy(t, np.random.default_rng(0), batch=3)
    for q in (0, 1):
        reach = t.reach(sigma, q)
        np.testing.assert_allclose(reach[:, 0], 1.0)
        for n in range(t.num_nodes):
            for a in np.flatnonzero(t.legal[n]):
                step = sigma[:, t.dec_index[n], a] if t.player[n] == q else 1.0
                np.testing.assert_allclose(reach[:, t.child[n, a]], reach[:, n] * step, atol=1e-15)


def test_normalise_and_bayes_update():
    np.testing.assert_allclose(normalise(np.array([1.0, 3.0, 0.0])), [0.25, 0.75, 0.0], atol=1e-70)
    np.testing.assert_allclose(normalise(np.zeros(4)), 0.25)  # the official 1e-80: a range without mass becomes uniform
    np.testing.assert_allclose(normalise(np.zeros(4), np.array([1.0, 0.0, 1.0, 0.0])), [0.5, 0.0, 0.5, 0.0])  # never a blocked hand
    np.testing.assert_allclose(normalise(np.array([1.0, 3.0, 4.0]), np.array([1.0, 1.0, 0.0])), [0.25, 0.75, 0.0])
    belief, probs = np.array([0.5, 0.3, 0.2]), np.array([0.1, 1.0, 0.0])
    np.testing.assert_allclose(bayes_update(belief, probs), [0.05 / 0.35, 0.3 / 0.35, 0.0], atol=1e-70)
    np.testing.assert_allclose(bayes_update(belief, np.zeros(3)), 1 / 3)  # an action the policy never takes


def test_ranges_after_actions_and_the_board_card_match_brute_force():
    """The PBS bookkeeping against the joint distribution of (hands, betting, board card) enumerated with the game."""
    g = make_game("leduc")
    s = _leduc()
    t0 = s.tree0
    rng = np.random.default_rng(0)
    sigma = _strategy(t0, rng)
    root = normalise(rng.random((1, 2, H)))
    beliefs = s.solver(root).leaf_beliefs(sigma)[0]  # (leaf, player, hand)
    joint = np.zeros((len(t0.leaves), H, H, H))  # [leaf, board card, player 0's hand, player 1's hand]
    for h0, h1 in itertools.permutations(range(H), 2):
        for leaf, n in enumerate(t0.leaves):
            st = g.new_initial_state()
            st.apply(h0)
            st.apply(h1)
            p = root[0, 0, h0] * root[0, 1, h1]
            for a in t0.hist[n][0]:
                node = t0.index[tuple(tuple(x) for x in st.history)]
                p *= sigma[0, t0.dec_index[node], a, st.cards[st.current_player]]
                st.apply(a)
            for card, prob in st.chance_outcomes():
                joint[leaf, card, h0, h1] = p * prob
    for leaf in range(len(t0.leaves)):
        np.testing.assert_allclose(board_distribution(beliefs[leaf]), joint[leaf].sum((1, 2)) / joint[leaf].sum(), atol=1e-12)
        for card in range(H):
            post = deal_board(beliefs[leaf], card)
            assert post[0, card] == 0 and post[1, card] == 0
            np.testing.assert_allclose(post.sum(1), 1.0)
            pairs = np.outer(post[0], post[1]) * (1 - np.eye(H))  # the PBS's joint: independent ranges, distinct cards
            np.testing.assert_allclose(pairs / pairs.sum(), joint[leaf, card] / joint[leaf, card].sum(), atol=1e-12)
    assert np.abs(board_distribution(beliefs) - 1 / H).max() > 0.01  # not uniform: the ranges tell which cards are out
    np.testing.assert_allclose(deal_board(beliefs, np.arange(5))[np.arange(5), :, np.arange(5)], 0.0)  # one card per PBS


def test_profile_values_match_the_game():
    """Terminal payoffs with card removal, leaf value x opponent reach, the chance average and the info keys: the
    root values of a random profile computed through the round trees give the game's expected value."""
    g = make_game("leduc")
    s = _leduc()
    rng = np.random.default_rng(1)
    t0, t1, L = s.tree0, s.tree1, 5
    sigma0, sigma1 = _strategy(t0, rng), _strategy(t1, rng, L * H)  # the last round's rows: leaf x board card
    policy = TabularPolicy(g, s.table(sigma0[0], sigma1.reshape(L, H, *sigma1.shape[1:])))
    assert len(policy.table) == 936  # every infoset of the game
    want = expected_value(g, policy)

    def continuation(p, beliefs):  # the profile's value at the leaves: the last round under sigma1, averaged over the card
        leaf, board = np.repeat(np.arange(L), H), np.tile(np.arange(H), L)
        after = s.solver(deal_board(np.repeat(beliefs[0], H, axis=0), board), leaf, board).values(p, sigma1)[:, 0]
        return chance_average(after.reshape(L, H, H), beliefs[0, :, 1 - p])[None]

    sub = s.solver(s.root_beliefs, leaf_fn=continuation)
    for p, sign in ((0, 1.0), (1, -1.0)):
        root = sub.values(p, sigma0)[0, 0]  # per hand, against the opponent's range (1/6 each, one card blocked)
        assert root.mean() * 6 / 5 == pytest.approx(sign * want, abs=1e-12)
    k = ReBeL(make_game("kuhn"), iters=8)
    sigma = _strategy(k.tree0, rng, hands=3)
    want = expected_value(k.game, TabularPolicy(k.game, k.table(sigma[0])))
    assert k.solver(k.root_beliefs).values(0, sigma)[0, 0].mean() * 3 / 2 == pytest.approx(want, abs=1e-12)


def test_chance_average_weights_each_card_by_the_surviving_opponent_mass():
    values = np.zeros((3, 3))
    values[1, 0], values[2, 0], values[0, 0] = 2.0, 4.0, 99.0  # hand 0's value after card 1 / card 2 (card 0: its own)
    opp = np.array([0.5, 0.3, 0.2])
    # one card is left for the board once both hands are dealt: v(0) = mass(1) v_1(0) + mass(2) v_2(0)
    assert chance_average(values, opp)[0] == pytest.approx(0.7 * 2.0 + 0.8 * 4.0)


# ----------------------------------------------------------------------------- the solver
@pytest.mark.parametrize("n", [1, 2, 7, 40])
def test_solver_is_linear_cfr_on_kuhn(n):
    """One round of Kuhn is the whole game: with textbook regret matching the solver's average strategy after n updates
    per player is the average of the repo's tabular Linear CFR after n + 1 iterations (it counts the uniform strategy
    as iteration 1; the official solver has it in the average from the start)."""
    g = make_game("kuhn")
    k = ReBeL(g, iters=8, regret_floor=0.0)
    sub = k.solver(k.root_beliefs).run(2 * n)
    assert sub.floor == 0.0
    mine, ref = k.table(sub.average()[0]), CFR(g, "lcfr").iterate(n + 1).average_policy().table
    assert set(mine) == set(ref) and len(ref) == 12
    for key, probs in ref.items():
        np.testing.assert_allclose(mine[key], probs, atol=1e-9)


def test_regret_floor_is_the_official_smoothing():
    """The official solver floors regrets at 1e-80: where nothing is positive the strategy is uniform, and an action
    with non-positive regret keeps a probability of that order instead of exactly 0."""
    k = ReBeL(make_game("kuhn"), iters=8)
    official, textbook = k.solver(k.root_beliefs).run(1), CFRD(k.tree0, k.root_beliefs, k.show[-1], floor=0.0).run(1)
    assert official.floor == EPS
    for floor in (EPS, 0.0):  # the setting reaches both rounds' solvers
        s = _leduc(regret_floor=floor)
        last = s.solver(deal_board(s.root_beliefs, 0), np.array([0]), np.array([0]))
        assert s.solver(s.root_beliefs).floor == floor and last.floor == floor
    root = official.last[0, 0]  # player 0's first decision after one update: the bet has the positive regret
    np.testing.assert_allclose(root, textbook.last[0, 0], atol=1e-60)
    assert (root[1] > 0).all() and (root[1] < 1e-70).all() and (textbook.last[0, 0, 1] == 0).all()
    np.testing.assert_allclose(official.last[0, 1], np.broadcast_to(k.tree0.uniform[1], (3, 3)))  # player 1 has not moved yet


def test_root_values_are_averaged_with_linear_weights():
    s = _leduc()
    rng = np.random.default_rng(2)
    leaf, board = np.array([1, 4]), np.array([2, 5])
    sub = s.solver(deal_board(normalise(rng.random((2, 2, H))), board), leaf, board, record=True)
    seen = [[], []]
    for step in range(12):
        p = step % 2
        before = sub.values(p, sub.history[step])[:, 0]
        sub.step()
        np.testing.assert_allclose(sub.root_value[p], before, atol=1e-12)  # the value of the iterate the step started from
        seen[p].append(before)
    for p in (0, 1):
        w = np.arange(1.0, 7.0)  # the k-th update of a player counts k + 1
        np.testing.assert_allclose(sub.root_mean[p], np.tensordot(w, np.stack(seen[p]), 1) / w.sum(), atol=1e-12)
        assert np.abs(sub.root_mean[p] - seen[p][-1]).max() > 0.05  # neither the last iterate's values
        assert np.abs(sub.root_mean[p] - np.mean(seen[p], axis=0)).max() > 0.01  # nor their plain mean
    assert sub.updates == [6, 6] and sub.steps == 12 and len(sub.history) == 13
    np.testing.assert_allclose(sub.root_mean[:, np.arange(2), :, ][:, np.arange(2), board], 0.0)  # the blocked hand has no value


def test_leaf_query_uses_the_current_iterate_and_scales_by_opponent_reach():
    s = _leduc()
    t0, rng = s.tree0, np.random.default_rng(3)
    L = len(t0.leaves)
    answers = rng.normal(size=(2, L, H))
    asked = []

    def leaf_fn(p, beliefs):
        asked.append((p, beliefs.copy()))
        return answers[p][None]

    root = normalise(rng.random((1, 2, H)))
    sub = s.solver(root, leaf_fn=leaf_fn, record=True)
    for step in range(6):
        sub.step()
        iterate = sub.history[step]  # the strategies the step started from - not the average, not the updated ones
        reach = [root[:, q, None, :] * t0.reach(iterate, q) for q in (0, 1)]
        p, beliefs = asked[step]
        assert p == step % 2 and beliefs.shape == (1, L, 2, H)
        for q in (0, 1):
            np.testing.assert_allclose(beliefs[0, :, q], normalise(reach[q][0, t0.leaves]), atol=1e-12)
        np.testing.assert_allclose(beliefs.sum(-1), 1.0)
        values = sub.values(p, iterate)  # asks again: the same iterate, the same answer
        want = answers[p] * reach[1 - p][0, t0.leaves].sum(1, keepdims=True)  # x the OPPONENT's total reach of the leaf
        np.testing.assert_allclose(values[0, t0.leaves], want, atol=1e-12)
        asked.pop()
    average = sub.average()
    assert np.abs(sub.leaf_beliefs(average) - sub.leaf_beliefs(sub.history[5])).max() > 0.01  # the average would be other PBSs
    assert np.abs(reach[0][0, t0.leaves].sum(1) - reach[1][0, t0.leaves].sum(1)).max() > 0.01  # the two reaches differ


def test_exact_leaf_values_reproduce_the_calibration():
    """Search alone, with the rest of the game solved exactly at every leaf query: the random-iterate mixture is close
    to full-game Linear CFR at the same number of updates (139.5 mA/g at 32 per player) and unsafe search is worse."""
    s = _leduc(iters=64)
    search = s.search(leaf_fn=s.exact_leaf(64))
    out = s.policies(search)
    exact, unsafe = (exploitability(s.game, out[k])[0] for k in ("exact", "unsafe"))
    assert 0.10 < exact < 0.15 and exact < unsafe < 0.19  # 0.127 and 0.143; the brief's prototype: 0.135 and 0.166
    assert -0.25 < s.root_value(search) < -0.05  # the game's value is -0.0856; the linear average approaches it slowly
    assert s.root_value(search) == pytest.approx(search.root_mean[0, 0].mean() * 6 / 5)  # 5 of 6 opponent cards are possible
    assert out["value_error"] is None and "sampled" not in out


# ----------------------------------------------------------------------------- stopping steps, sampling, examples
def test_stopping_step_distributions():
    np.testing.assert_allclose(stop_weights(8, "uniform"), np.full(9, 1 / 9))  # training: every step, the last included
    np.testing.assert_allclose(stop_weights(8, "linear"), np.array([1, 0, 2, 0, 3, 0, 4, 0, 0]) / 10)  # play: the average's weights
    s = _leduc(iters=8, games=20_000, leaf_targets="solve", chance_prob=0.0)
    search = s.search()
    freq = np.bincount(s.self_play(search)["stop"], minlength=9) / 20_000
    assert np.abs(freq - 1 / 9).max() < 0.012  # the official training loop: uniform over 0 .. T
    s.train_stop = "linear"
    freq = np.bincount(s.self_play(search)["stop"], minlength=9) / 20_000
    assert np.abs(freq - stop_weights(8, "linear")).max() < 0.012
    # test time: playthroughs and the sampled mixture both use the average strategy's weights
    rng = np.random.default_rng(0)
    plays = [s.playthrough(rng, search) for _ in range(4000)]
    freq = np.bincount([p.stop0 for p in plays], minlength=9) / 4000
    assert np.abs(freq - stop_weights(8, "linear")).max() < 0.025
    out = s.policies(search, samples=600, rng=rng)
    assert out["stop0"].shape == (600,) and out["stop1"].shape == (600, 5, H)
    for stops in (out["stop0"], out["stop1"]):
        assert np.abs(np.bincount(stops.ravel(), minlength=9) / stops.size - stop_weights(8, "linear")).max() < 0.05
    same = (out["stop1"] == out["stop0"][:, None, None]).mean()
    assert 0.2 < same < 0.4  # every subgame draws its own step: sum of the squared weights = 0.3


def test_exploration_is_by_exactly_one_player():
    """With epsilon = 1 and an iterate that always calls: the explorer acts at random, the other player never
    deviates - and the ranges follow the ITERATE, which gives a raise no mass (the range becomes uniform)."""
    t0 = _leduc().tree0
    rng = np.random.default_rng(4)
    G = 4000
    sigma = np.zeros((1, t0.num_decisions, 3, H))
    sigma[:, :, 1] = 1.0
    root = normalise(rng.random((1, 2, H)))
    node, beliefs, explorer = sample_leaf(t0, np.broadcast_to(sigma, (G,) + sigma.shape[1:]), np.repeat(root, G, axis=0), 1.0, rng)
    reached = [{t0.hist[n][0] for n in np.unique(node[explorer == i])} for i in (0, 1)]
    assert reached[0] == {(1, 1), (2, 1)}  # player 0 explores: check or bet, player 1 always calls
    assert reached[1] == {(1, 1), (1, 2, 1)}  # player 1 explores after the check, player 0 always calls
    assert 0.45 < explorer.mean() < 0.55
    for i in (0, 1):
        raised = (explorer == i) & (node != t0.index[((1, 1), ())])
        assert 0.4 < raised[explorer == i].mean() < 0.6  # epsilon = 1: uniform over the two legal actions
        np.testing.assert_allclose(beliefs[raised][:, i], 1 / H)  # the iterate never raises: no mass -> uniform
        np.testing.assert_allclose(beliefs[raised][:, 1 - i], np.broadcast_to(root[0, 1 - i], (raised.sum(), H)))
    quiet = node == t0.index[((1, 1), ())]
    np.testing.assert_allclose(beliefs[quiet], np.broadcast_to(root, (quiet.sum(), 2, H)))
    node, _, _ = sample_leaf(t0, np.broadcast_to(sigma, (G,) + sigma.shape[1:]), np.repeat(root, G, axis=0), 0.0, rng)
    assert {t0.hist[n][0] for n in np.unique(node)} == {(1, 1)}  # no exploration: the iterate's line only


@pytest.mark.parametrize("epsilon", [0.0, 0.5])
def test_sampled_leaves_follow_the_iterate(epsilon):
    """Without exploration the public line has the probability the joint distribution of the two hands gives it
    (hands exclude each other); with or without it the ranges at the node reached are the iterate's."""
    s = _leduc()
    t0, rng = s.tree0, np.random.default_rng(5)
    G = 60_000
    sigma = _strategy(t0, rng) ** 3
    sigma /= sigma.sum(2, keepdims=True)
    root = np.array([[[0.45, 0.45, 0.02, 0.02, 0.03, 0.03], [0.6, 0.02, 0.3, 0.02, 0.03, 0.03]]])  # they often block each other
    node, beliefs, _ = sample_leaf(t0, np.broadcast_to(sigma, (G,) + sigma.shape[1:]), np.repeat(root, G, axis=0), epsilon, rng)
    reach = [root[0, q] * t0.reach(sigma, q)[0] for q in (0, 1)]  # (node, hand)
    ends = np.flatnonzero(t0.kind != DECISION)
    assert set(np.unique(node)) <= set(ends)
    for n in ends:
        at = node == n
        np.testing.assert_allclose(beliefs[at], np.broadcast_to(np.stack([normalise(reach[q][n]) for q in (0, 1)]), (at.sum(), 2, H)),
                                   atol=1e-12)
    if epsilon == 0:
        pairs = np.array([reach[0][n].sum() * reach[1][n].sum() - reach[0][n] @ reach[1][n] for n in range(t0.num_nodes)])
        freq = np.array([(node == n).mean() for n in ends])
        assert pairs[ends].sum() == pytest.approx(pairs[0])
        assert np.abs(freq - pairs[ends] / pairs[0]).max() < 0.008
        naive = np.array([reach[0][n].sum() * reach[1][n].sum() for n in ends])  # independent hands: measurably different
        assert np.abs(naive / naive.sum() - pairs[ends] / pairs[0]).max() > 0.03


def test_board_card_is_drawn_from_the_ranges(monkeypatch):
    rng = np.random.default_rng(6)
    beliefs = np.array([[0.5, 0.3, 0.1, 0.05, 0.03, 0.02], [0.02, 0.6, 0.1, 0.1, 0.1, 0.08]])
    cards = sample_board(np.broadcast_to(beliefs, (40_000, 2, H)), rng)
    assert np.abs(np.bincount(cards, minlength=H) / 40_000 - board_distribution(beliefs)).max() < 0.008
    assert board_distribution(beliefs).max() - board_distribution(beliefs).min() > 0.1
    # in self-play: player 0 surely holds card 0 and player 1 card 1 at the leaf, so the board is one of the other four
    import headsup.algos.rebel as R

    s = _leduc(iters=8, games=300)
    sure = np.zeros((300, 2, H))
    sure[:, 0, 0] = sure[:, 1, 1] = 1.0
    leaf = s.tree0.leaves[np.arange(300) % 5]
    monkeypatch.setattr(R, "sample_leaf", lambda tree, sigma, beliefs, eps, rng: (leaf, sure, np.zeros(300, dtype=int)))
    out = s.self_play(s.search())
    cards = out["board"][out["stage"] == AFTER_BOARD]
    assert len(cards) == 300 and set(cards.tolist()) == {2, 3, 4, 5}
    np.testing.assert_allclose(out["beliefs"][out["stage"] == AFTER_BOARD], sure, atol=1e-70)


@pytest.mark.parametrize("mode", ["net", "solve"])
def test_self_play_stores_the_averaged_root_values_of_every_pbs(mode):
    s = _leduc(iters=16, games=48, leaf_targets=mode, seed=3)
    with torch.no_grad():  # an untrained network answers almost 0: give it something to say
        s.net[-1].weight *= 30
    search = s.search()
    out = s.self_play(search)
    stage, leaf, board, beliefs, values = (out[k] for k in ("stage", "leaf", "board", "beliefs", "values"))
    at_leaf = s.tree0.kind[out["node"]] == LEAF
    assert 10 < at_leaf.sum() < 48 and len(out["stop"]) == 48
    # the root PBS: the root solve's average of the iterates' root values
    assert stage[0] == ROOT and (stage[1:] != ROOT).all()
    np.testing.assert_allclose(values[0], search.root_mean[:, 0], atol=1e-12)
    np.testing.assert_allclose(beliefs[0], 1 / H)
    after, before = np.flatnonzero(stage == AFTER_BOARD), np.flatnonzero(stage == BEFORE_BOARD)
    assert len(before) == at_leaf.sum() and (board[before] == -1).all() and len(stage) == 1 + len(after) + len(before)
    if mode == "net":
        # after the card: the last round's solve from the ranges without the card; the target is its root_mean ...
        assert len(after) == at_leaf.sum() and (leaf[after] == leaf[before]).all()
        np.testing.assert_allclose(beliefs[after], deal_board(beliefs[before], board[after]), atol=1e-12)
        sub = s.solver(beliefs[after], leaf[after], board[after]).run(16)
        np.testing.assert_allclose(values[after], sub.root_mean.transpose(1, 0, 2), atol=1e-12)
        assert np.abs(values[after] - sub.root_value.transpose(1, 0, 2)).max() > 0.05  # ... not the last iterate's values
        # before the card: the network's values after every card, averaged with card removal (the paper's poker agent)
        want = np.zeros((len(before), 2, H))
        for i, k in enumerate(before):
            for p in (0, 1):
                per_card = np.stack([s.net_values(s.public(s.tree0.pot[s.tree0.leaves][leaf[k]], np.array([c])),
                                                  deal_board(beliefs[k], c)[None], p)[0] for c in range(H)])
                want[i, p] = chance_average(per_card, beliefs[k, 1 - p])
        np.testing.assert_allclose(values[before], want, atol=1e-5)
        assert np.abs(want).max() > 0.05
    else:  # exact targets: every card's subgame solved; nothing is stored after the card
        assert len(after) == 0
        np.testing.assert_allclose(values[before], s.solve_boards(leaf[before], beliefs[before], 16)[0], atol=1e-12)
    # the examples: one row per PBS and agent, targets in units of the largest pot
    x, y = s.examples_of(out)
    M = len(stage)
    assert x.shape == (2 * M, s.in_dim) and y.shape == (2 * M, H) and x.dtype == np.float32 and y.dtype == np.float32
    for agent in (0, 1):
        rows = slice(agent * M, (agent + 1) * M)
        np.testing.assert_allclose(y[rows], values[:, agent] / 13.0, atol=1e-6)
        assert (x[rows, 0] == agent).all()
        np.testing.assert_array_equal(x[rows, 1], stage == BEFORE_BOARD)
        np.testing.assert_allclose(x[rows, 2], np.where(stage == ROOT, 2, s.tree0.pot[s.tree0.leaves][leaf]) / 26.0, atol=1e-7)
        np.testing.assert_array_equal(x[rows, 3:3 + H].argmax(1)[board >= 0], board[board >= 0])
        np.testing.assert_array_equal(x[rows, 3:3 + H].sum(1), board >= 0)
        np.testing.assert_allclose(x[rows, 3 + H:], beliefs.reshape(M, 2 * H), atol=1e-7)


def test_chance_prob_thins_the_examples_before_the_card():
    s = _leduc(iters=8, games=400, chance_prob=1 / 3, seed=1)
    out = s.self_play(s.search())
    n_after, n_before = (out["stage"] == AFTER_BOARD).sum(), (out["stage"] == BEFORE_BOARD).sum()
    assert n_after > 150 and 0.2 < n_before / n_after < 0.47  # the paper: one in three


def test_last_round_uses_no_network():
    s = _leduc(iters=8, games=64, leaf_targets="solve")
    search = s.search()
    asked = []
    s.net_values = lambda *a: asked.append(a) or (_ for _ in ()).throw(AssertionError("the network was asked"))
    out = s.self_play(search)  # exact targets: the last round's solves, never the network
    assert (out["stage"] == BEFORE_BOARD).sum() > 10 and not asked
    assert len(s.tree1.leaves) == 0 and s.solver(deal_board(s.root_beliefs, 0), np.array([0]), np.array([0])).leaf_fn is None
    s = _leduc(iters=8, games=64, leaf_targets="net")
    search, calls, real = s.search(), [], s.net_values
    s.net_values = lambda pub, beliefs, agent: calls.append((pub, agent)) or real(pub, beliefs, agent)
    out = s.self_play(search)
    # the paper's targets before the card: one batch per agent, at PBSs AFTER a card - and that is all
    assert [agent for _, agent in calls] == [0, 1]
    assert all((pub[:, 0] == 0).all() and (pub[:, 2:].sum(1) == 1).all() for pub, _ in calls)
    assert len(calls[0][0]) == H * (out["stage"] == BEFORE_BOARD).sum()


def test_generate_fills_the_buffer_once_per_root_solve():
    s = _leduc(iters=8, games=40, capacity=64)
    searches = []
    solve = s.search
    s.search = lambda *a, **k: searches.append(1) or solve(*a, **k)
    out = s.generate()
    assert len(searches) == 1  # the root PBS is the same for every game: one solve serves the batch
    n = 2 * len(out["stage"])
    assert s.examples == n and s.games_played == 40 and s.buffer.size == min(n, 64)
    x, y = s.examples_of(out)
    last = (s.buffer.pos - 1) % 64
    np.testing.assert_array_equal(s.buffer.x[last], x[-1])  # FIFO: the newest example was written last
    np.testing.assert_array_equal(s.buffer.y[last], y[-1])
    buf = Replay(3, 2, 1)
    buf.add(np.arange(8.0).reshape(4, 2), np.arange(4.0).reshape(4, 1))
    assert buf.size == 3 and buf.pos == 1 and buf.y[:, 0].tolist() == [3.0, 1.0, 2.0]


# ----------------------------------------------------------------------------- the network
def test_value_net_and_loss():
    torch.manual_seed(0)
    net = value_net(21, 6)
    kinds = [type(m).__name__ for m in net]
    assert kinds == ["Linear", "LayerNorm", "GELU", "Linear", "LayerNorm", "GELU", "Linear"]  # the official Net2
    assert [m.out_features for m in net if isinstance(m, torch.nn.Linear)] == [256, 256, 6]
    assert float(net[-1].weight.detach().abs().max()) <= 0.01 / 16 + 1e-9  # the output layer is scaled by 0.01: |w| <= 1 / sqrt(256)
    with torch.no_grad():
        assert float(net(torch.randn(50, 21)).abs().max()) < 0.02
    x = torch.tensor([[0.5, -0.5, 2.0, -3.0, 0.0, 1.0]])
    assert huber(x).tolist() == [[0.25, 0.25, 3.0, 5.0, 0.0, 1.0]]  # x^2 up to 1, then 2 |x| - 1: twice the textbook Huber
    s = _leduc()
    with torch.no_grad():
        s.net[-1].weight.zero_()
        s.net[-1].bias.copy_(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    y = torch.zeros(4, H)
    y[0, 0] = 4.0
    want = ((0.25 + 1.0 + 2.0 + 3.0 + 4.0) * 4 + 7.0) / 24  # mean over hands and examples
    assert s.loss(torch.zeros(4, s.in_dim), y).item() == pytest.approx(want, abs=1e-6)


def test_training_steps_follow_the_official_schedule():
    s = _leduc(lr=3e-4, lr_halve_every=400, steps=3, batch=16)
    assert [s.learning_rate(e) for e in (0, 399, 400, 799, 800, 5000)] == [3e-4, 3e-4, 1.5e-4, 1.5e-4, 7.5e-5, 7.5e-5]
    s.generate()
    before = [p.detach().clone() for p in s.net.parameters()]
    s.epoch = 400
    s.train(3)
    assert s.sgd_steps == 3 and s.opt.param_groups[0]["lr"] == 1.5e-4 and s.last_loss is not None
    assert all((a != b).any() for a, b in zip(before, s.net.parameters()))
    assert int(next(iter(s.opt.state.values()))["step"]) == 3 and s.grad_clip == 5.0
    assert ReBeL(make_game("kuhn"), iters=8).train(2) is None  # nothing to train on yet


def test_network_input_and_units():
    s = _leduc()
    pub = s.public(np.array([6.0, 10.0]), np.array([4, 0]))
    assert pub.tolist() == [[0, 6 / 26, 0, 0, 0, 0, 1, 0], [0, 10 / 26, 1, 0, 0, 0, 0, 0]]
    assert s.leaf_pub[:, 0].tolist() == [1] * 5 and s.leaf_pub[:, 1].tolist() == [p / 26 for p in (2, 6, 10, 6, 10)]
    beliefs = np.arange(24.0).reshape(2, 2, H)
    x = s.encode(pub, beliefs, 1)
    assert x.shape == (2, 21) and x[:, 0].tolist() == [1, 1] and x[1, 9:].tolist() == list(range(12, 24))
    with torch.no_grad():
        s.net[-1].weight.zero_()
        s.net[-1].bias.fill_(0.5)
    np.testing.assert_allclose(s.net_values(pub, beliefs, 0), 6.5)  # the network works in units of the largest pot (13)
    np.testing.assert_allclose(s.leaf_values(1, np.full((3, 5, 2, H), 1 / H)), 6.5)


# ----------------------------------------------------------------------------- play and evaluation
def test_exact_mixture_is_the_expectation_of_the_playthroughs():
    """policies()['exact'] against brute force: every pair of stopping steps played as a Playthrough (which solves the
    last round on demand) and mixed over the game tree with the test-time weights."""
    s = _leduc(iters=6)
    g = s.game
    search = s.search(leaf_fn=s.exact_leaf(4))
    w = stop_weights(6, "linear")
    pairs = [(a, b) for a in (0, 2, 4) for b in (0, 2, 4)]
    plays = [Playthrough(s, search, stop0=a, stop1=b) for a, b in pairs]
    want = _mix(g, plays, [w[a] * w[b] for a, b in pairs])
    got = s.policies(search)["exact"].table
    assert len(want) > 500
    for key, probs in want.items():
        np.testing.assert_allclose(got[key], probs, atol=1e-10, err_msg=str(key))
    # one stopping step per subgame, for all its decisions: the last round's strategy is that iterate of that solve
    play = plays[5]  # stops (2, 4)
    assert play.stop0 == 2 and all(stop == 4 for stop, _ in play.round1.values()) and len(play.round1) == 30
    beliefs = search.leaf_beliefs(search.history[2])[0]
    for (leaf, card), (_, sigma) in list(play.round1.items())[::7]:
        sub = s.solver(deal_board(beliefs[leaf][None], card), np.array([leaf]), np.array([card])).run(4)
        np.testing.assert_allclose(sigma, sub.last[0], atol=1e-12)


def test_sampled_mixture_averages_its_playthroughs():
    s = _leduc(iters=6)
    search = s.search(leaf_fn=s.exact_leaf(4))
    out = s.policies(search, samples=4, rng=np.random.default_rng(1))
    plays = [Playthrough(s, search, stop0=out["stop0"][k], stop1=out["stop1"][k]) for k in range(4)]
    want = _mix(s.game, plays, np.ones(4))
    got = out["sampled"].table
    for key, probs in want.items():
        np.testing.assert_allclose(got[key], probs, atol=1e-10, err_msg=str(key))
    assert len(set(out["stop0"].tolist())) > 1 and len(np.unique(out["stop1"])) > 1


def test_unsafe_policy_is_the_average_strategy_with_its_own_ranges():
    """The comparison policy: the root solve's average strategy, and below every leaf and card the last round's
    average after T steps from the ranges the AVERAGE reaches the leaf with."""
    s = _leduc(iters=8)
    search = s.search(leaf_fn=s.exact_leaf(4))
    table = s.policies(search)["unsafe"].table
    average = search.average()
    t0, t1 = s.tree0, s.tree1
    for d, n in enumerate(t0.dec):
        for h in range(H):
            np.testing.assert_allclose(table[(int(t0.player[n]), h, None, 0, t0.hist[n])], average[0, d, :, h], atol=1e-12)
    beliefs = search.leaf_beliefs(average)[0]
    checked = 0
    for leaf, card in ((0, 1), (2, 4), (4, 0)):
        sub = s.solver(deal_board(beliefs[leaf][None], card), np.array([leaf]), np.array([card])).run(8)
        tree, sigma = s.trees1[leaf], sub.average()[0]
        for d, n in enumerate(tree.dec):
            for h in range(H):
                key = (int(t1.player[n]), h, card, 1, tree.hist[n])
                if key in table:  # hands the average strategy brings here
                    np.testing.assert_allclose(table[key], sigma[d, :, h], atol=1e-12)
                    checked += 1
    assert checked > 60
    assert np.abs(sub.average() - s.solver(sub.beliefs, np.array([4]), np.array([0])).run(4).average()).max() > 0.01


def test_playthrough_draws_one_stop_per_subgame_and_keeps_it():
    s = _leduc(iters=8)
    search = s.search()
    g = s.game
    play = s.playthrough(np.random.default_rng(0), search)
    st = g.new_initial_state()
    for a in (3, 1, 1, 1, 5):  # deal, check, check, board card
        st.apply(a)
    first = play(st).copy()
    assert first.shape == (3,) and first.sum() == pytest.approx(1.0) and len(play.round1) == 1
    stop, sigma = play.round1[(0, 5)]
    assert stop % 2 == 0 and stop < 8
    st.apply(2)
    np.testing.assert_array_equal(play(st), sigma[s.tree1.dec_index[s.tree1.index[((1, 1), (2,))]], :, 1])
    assert len(play.round1) == 1 and play.round1[(0, 5)][1] is sigma  # the same solve, the same stopping step
    root = g.new_initial_state()
    root.apply(3)
    root.apply(1)
    np.testing.assert_array_equal(play(root), search.history[play.stop0][0, 0, :, 3])
    # two playthroughs play a hand against each other through the game protocol
    rng = np.random.default_rng(1)
    for _ in range(5):
        end = g.playout([s.playthrough(rng, search), s.playthrough(rng, search)], rng)
        assert end.is_terminal() and abs(end.returns()[0]) <= 13


def test_kuhn_is_solved_without_a_network():
    """One round: the subgame is the whole game, the policy played is Linear CFR's average."""
    g = make_game("kuhn")
    s = ReBeL(g, iters=64, games=16, steps=2, batch=8, eval_samples=64)
    ev = s.evaluate()
    want = exploitability(g, TabularPolicy(g, s.table(s.solver(s.root_beliefs).run(62).average()[0])))[0]
    assert ev["exploitability"] == pytest.approx(want, abs=1e-12) and ev["exploitability"] < 0.01
    assert ev["exploitability"] < ev["exploitability_sampled"] < 0.05 and ev["samples"] == 64  # 64 sampled iterates: an upper bound
    assert ev["value_error"] is None and ev["value_error_search"] is None
    assert ev["root_value"] == pytest.approx(-1 / 18, abs=0.03)  # the linear average of the iterates' values converges slowly
    out = s.generate()
    assert out["stage"].tolist() == [ROOT] and s.examples == 2  # the root PBS is the only one there is


def test_evaluation_reports_and_does_not_disturb_training():
    def run(evaluate):
        s = _leduc(iters=8, games=24, seed=5)
        s.iterate(1)
        ev = s.evaluate() if evaluate else None
        s.iterate(1)
        return ev, torch.cat([p.detach().flatten() for p in s.net.parameters()])

    (ev, a), (_, b) = run(True), run(False)
    torch.testing.assert_close(a, b)  # the stopping steps of an evaluation come from their own generator
    assert set(ev) == {"exploitability", "exploitability_sampled", "samples", "exploitability_unsafe", "root_value", "value_error",
                       "value_error_search"}
    assert 0 < ev["exploitability"] < 3 and 0 < ev["exploitability_sampled"] < 3 and ev["samples"] == 8
    assert ev["value_error"] > 0 and ev["value_error_search"] > 0
    # the sampled mixture: the configured number of playthroughs at the evaluation's search length, other draws at every epoch
    s = _leduc(iters=8, eval_iters=6, eval_samples=5, seed=5)
    seen, policies = [], s.policies
    s.policies = lambda search, k, rng, **kw: seen.append((search.steps, k, rng.random())) or policies(search, k, rng, **kw)
    for epoch in (3, 3, 4):
        s.epoch = epoch
        s.evaluate()
    assert [x[:2] for x in seen] == [(6, 5)] * 3 and seen[0][2] == seen[1][2] != seen[2][2]


def test_value_error_measures_the_network_against_exact_solves():
    s = _leduc(iters=8, probe_iters=8)
    beliefs, values, w = s.probe()
    assert beliefs.shape == (4, 5, 2, H) and values.shape == (20, 2, H) and w.tolist() == [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_allclose(values, s.solve_boards(np.tile(np.arange(5), 4), beliefs.reshape(20, 2, H), 8)[0], atol=1e-12)
    with torch.no_grad():
        s.net[-1].weight.zero_()
        s.net[-1].bias.zero_()
    assert s.value_error() == pytest.approx(np.sqrt(w @ (values ** 2).reshape(4, -1).mean(1)))  # RMS in chips, iterate weights
    search = s.search()
    # at the search's own leaf PBSs (every even iterate, solved exactly): the error of what the search was fed, value x opponent reach
    out = s.policies(search, value_error=True)
    iterates = np.concatenate(search.history)[0:8:2]
    exact = s.solve_boards(np.tile(np.arange(5), 4), search.leaf_beliefs(iterates).reshape(20, 2, H), 6)[0].reshape(4, 5, 2, H)
    fed = np.stack([search.reach(iterates, 1 - p)[:, s.tree0.leaves].sum(-1) for p in (0, 1)], axis=2)
    assert out["value_error"] == pytest.approx(np.sqrt(w @ ((exact * fed[..., None]) ** 2).reshape(4, -1).mean(1)))
    assert 0.1 < fed.max() <= 1.0 and fed.min() < 0.05


def test_training_lowers_the_value_error():
    s = _leduc(iters=16, games=96, steps=25, batch=128, lr=1e-3, probe_iters=16, seed=0)
    start = s.value_error()
    s.iterate(12)
    assert s.epoch == 12 and s.sgd_steps == 300 and s.examples > 1500
    assert s.value_error() < 0.5 * start


# ----------------------------------------------------------------------------- checkpoints and the CLI
def test_checkpoint_round_trip_continues_identically():
    a = _leduc(seed=2).iterate(2)
    blob = io.BytesIO()
    torch.save(a.state_dict(), blob)
    blob.seek(0)
    b = _leduc(seed=99).load_state_dict(torch.load(blob, weights_only=False))
    assert (b.epoch, b.examples, b.sgd_steps, b.games_played) == (2, a.examples, 4, 64)
    a.iterate(2)
    b.iterate(2)
    for x, y in zip(a.net.parameters(), b.net.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert a.examples == b.examples and a.buffer.pos == b.buffer.pos and a.last_loss == b.last_loss
    np.testing.assert_array_equal(a.buffer.x[: a.buffer.size], b.buffer.x[: b.buffer.size])
    np.testing.assert_array_equal(a.buffer.y[: a.buffer.size], b.buffer.y[: b.buffer.size])
    assert a.evaluate() == b.evaluate()
    with pytest.raises(ValueError, match="cannot be loaded"):
        ReBeL(make_game("kuhn"), iters=8).load_state_dict(a.state_dict())
    with pytest.raises(ValueError, match="capacity"):
        _leduc(capacity=8).load_state_dict(a.state_dict())


def test_state_dict_is_a_snapshot_and_solvers_never_share_optimiser_state():
    def steps(solver):
        return int(next(iter(solver.opt.state.values()))["step"])

    a = _leduc().iterate(1)
    snap = a.state_dict()
    weights = snap["net"]["0.weight"].clone()
    b = _leduc().load_state_dict(snap)
    a.iterate(1)
    assert int(snap["opt"]["state"][0]["step"]) == 2 and steps(a) == 4 and steps(b) == 2
    torch.testing.assert_close(snap["net"]["0.weight"], weights, rtol=0, atol=0)
    b.iterate(2)
    assert steps(a) == 4 and steps(b) == 6 and int(snap["opt"]["state"][0]["step"]) == 2
    assert snap["buffer"]["x"].base is None and len(snap["buffer"]["x"]) == snap["buffer"]["size"]  # a copy of the filled part


def test_cli_trains_evaluates_and_resumes(tmp_path, capsys):
    out, ck = str(tmp_path / "runs" / "k.json"), str(tmp_path / "k.pt")
    args = ["--game", "leduc", "--iters", "8", "--games", "24", "--steps", "2", "--batch", "16", "--buffer", "4096", "--eval-every", "2",
            "--eval-samples", "8", "--json", out, "--checkpoint", ck, "--seed", "1"]
    main(args + ["--epochs", "2"])
    first = json.load(open(out))
    assert first["game"] == "leduc" and first["algo"] == "rebel" and [c["epoch"] for c in first["curve"]] == [1, 2]
    row = first["curve"][-1]
    assert {"epoch", "examples", "sgd_steps", "games", "exploitability", "exploitability_sampled", "samples", "exploitability_unsafe",
            "root_value", "value_error", "value_error_search", "loss", "seconds"} == set(row)
    assert row["sgd_steps"] == 4 and row["games"] == 48 and row["examples"] > 20 and row["samples"] == 8
    main(args + ["--epochs", "4"])
    assert "resumed from" in capsys.readouterr().out
    second = json.load(open(out))
    assert [c["epoch"] for c in second["curve"]] == [1, 2, 4] and second["curve"][:2] == first["curve"]
    assert second["curve"][-1]["sgd_steps"] == 8 and second["curve"][-1]["seconds"] > row["seconds"]
    main(["--game", "kuhn", "--oracle", "--iters", "32", "--eval-samples", "16", "--json", out])
    oracle = json.load(open(out))
    assert oracle["algo"] == "rebel-oracle" and oracle["curve"][0]["iters"] == 32 and oracle["curve"][0]["exploitability"] < 0.03
    with pytest.raises(ValueError, match="small game"):
        main(["--game", "fhp", "--epochs", "1"])
