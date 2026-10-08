"""AlphaHoldem (Zhao et al., AAAI-22) on the default game: tensors, network, Trinal-Clip PPO, K-Best pool, trainer, player."""

import numpy as np
import pytest
import torch

from headsup.engine import HeadsUpPoker, legal_mask_from_obs
from headsup.game import DEFAULT_GAME, FHP, GameConfig

TREES = {
    "default": DEFAULT_GAME,
    "five sizes": GameConfig(bet_sizes=(0.5, 0.75, 1.0, 1.5, 2.0), mask_redundant=True),
    "cap 4": GameConfig(bet_sizes=(0.5, 1.0), raise_cap=4, mask_redundant=True),
}


def _random_hands(game, hands, seed):
    """Random hands on a Python engine.  Per decision: the actor's observation, the hand's true history so far as
    [(round, slot, seat, executed action, legal set at that decision)], and the engine's state there.  Per finished
    hand: both seats' terminal observations with the complete history.  The executed action comes from the engine's
    own bookkeeping (its twins and the chips that moved), not from the observation."""
    rng = np.random.default_rng(seed)
    e = HeadsUpPoker(game=game, rng=np.random.default_rng(seed + 1))
    decisions, terminals = [], []
    for _ in range(hands):
        e.reset()
        history, slots = [], [0] * 4
        while not e.done:
            legal, twins = e.legal_mask_and_twins()
            p, r, to_call, stack = e.current, int(e.stage), e.to_call, e.stacks[e.current]
            decisions.append(dict(obs=e.observation(), history=list(history), round=r, slot=slots[r], seat=p, legal=list(legal),
                                  hand=e.hands[p], board=list(e.visible_board), bets=(e.bets[p], e.bets[1 - p])))
            # every index, also masked ones; mostly check / call so that hands reach the river
            weights = np.array([0.3, 2.5] + [1.5 / game.num_raises] * game.num_raises + [0.3] * game.all_in)
            a = int(rng.choice(game.num_actions, p=weights / weights.sum()))
            before = e.bets[p]
            e.step(a)
            put = e.bets[p] - before
            executed = twins[a]
            if e.folded == p:
                executed = 0
            elif put == min(to_call, stack):
                executed = 1  # also an all-in for exactly the call
            elif game.all_in and put == stack:
                executed = game.all_in_action  # also a raise that the cap or the stack made an all-in
            history.append((r, slots[r], p, executed, list(legal)))
            slots[r] += 1
        for seat in (0, 1):
            terminals.append(dict(obs=e.observation(seat), history=list(history)))
    return decisions, terminals


def _expected_actions(game, history, pending=None):
    """The action tensor the paper describes, written down from the true history."""
    x = np.zeros((24, 4, game.num_actions), dtype=bool)
    for r, k, seat, action, legal in history:
        if k < 6:
            x[6 * r + k, seat, action] = True  # the acting seat's row
            x[6 * r + k, 2, action] = True  # the sum of the two
            x[6 * r + k, 3] = legal  # the legal actions at that decision
    if pending is not None and pending[1] < 6:
        x[6 * pending[0] + pending[1], 3] = pending[2]
    return x


@pytest.mark.parametrize("tree", list(TREES))
def test_action_tensor_matches_the_true_history(tree):
    from headsup.alphaholdem.encoding import Encoder

    game = TREES[tree]
    decisions, terminals = _random_hands(game, 300, seed=7)
    enc = Encoder(game)
    obs = np.stack([d["obs"] for d in decisions])
    cards, acts, legal = enc(obs)
    assert acts.shape == (len(obs), 24, 4, game.num_actions) and acts.dtype == torch.bool
    expected = np.stack([_expected_actions(game, d["history"], (d["round"], d["slot"], d["legal"])) for d in decisions])
    np.testing.assert_array_equal(acts.numpy(), expected)
    np.testing.assert_array_equal(legal.numpy(), np.array([d["legal"] for d in decisions]))
    np.testing.assert_array_equal(legal.numpy(), legal_mask_from_obs(obs, game))
    assert max(len(d["history"]) for d in decisions) >= 8 and expected[:, 18:].any()  # long hands, river actions
    assert (expected[:, :, :2].sum(axis=(1, 2, 3)) == [len(d["history"]) for d in decisions]).all()
    # the seat to act follows from the pending slot (rows are absolute: seat 0 = small blind)
    pending_slot = np.array([6 * d["round"] + d["slot"] for d in decisions])
    first_actor = np.where(pending_slot < 6, 0, 1)
    np.testing.assert_array_equal((first_actor + pending_slot % 6) % 2, [d["seat"] for d in decisions])
    # terminal observations (a one-seat env hands them out after an open-fold): history rows as recorded, no crash
    t_obs = np.stack([t["obs"] for t in terminals])
    _, t_acts, t_legal = enc(t_obs)
    t_expected = np.stack([_expected_actions(game, t["history"]) for t in terminals])
    np.testing.assert_array_equal(t_acts.numpy()[:, :, :3], t_expected[:, :, :3])
    assert t_expected[:, :, :2, 0].any() and t_legal.numpy().any(axis=1).all()  # folds are recorded; something is always legal


def test_card_tensor_and_value_bounds():
    from headsup.alphaholdem.encoding import Encoder

    decisions, _ = _random_hands(DEFAULT_GAME, 150, seed=3)
    enc = Encoder(DEFAULT_GAME)
    obs = np.stack([d["obs"] for d in decisions])
    cards, _, _ = enc(obs)
    assert cards.shape == (len(obs), 6, 4, 13) and cards.dtype == torch.bool
    expected = np.zeros((len(obs), 6, 52), dtype=bool)
    for i, d in enumerate(decisions):
        board = d["board"]
        expected[i, 0, list(d["hand"])] = True
        expected[i, 1, board[:3]] = True
        expected[i, 2, board[3:4]] = True
        expected[i, 3, board[4:5]] = True
        expected[i, 4, board] = True
        expected[i, 5, list(d["hand"]) + board] = True
    np.testing.assert_array_equal(cards.numpy().reshape(len(obs), 6, 52), expected)  # index = 13 * suit + rank = the card id
    assert cards[0, 0, decisions[0]["hand"][0] // 13, decisions[0]["hand"][0] % 13]  # rows are suits, columns ranks
    assert {int(c.sum()) for c in cards[:, 4]} == {0, 3, 4, 5}
    own, opp = enc.value_bounds(obs)
    np.testing.assert_array_equal(own.numpy(), [d["bets"][0] for d in decisions])
    np.testing.assert_array_equal(opp.numpy(), [d["bets"][1] for d in decisions])


@pytest.mark.parametrize("game", [*TREES.values(), FHP, GameConfig(stack_size=200, bet_sizes=("min", 0.33, 3.0), raise_cap=5, mask_redundant=True)],
                         ids=[*TREES, "fhp", "200 chips"])
def test_legal_mask_equals_the_engines(game):
    from headsup.alphaholdem.encoding import Encoder

    rng = np.random.default_rng(0)
    e = HeadsUpPoker(game=game, rng=np.random.default_rng(1))
    obs, legal = [], []
    for _ in range(200):
        e.reset()
        while not e.done:
            obs.append(e.observation())
            legal.append(e.legal_mask())
            e.step(int(rng.integers(1 if rng.random() < 0.8 else 0, game.num_actions)))
    obs = np.stack(obs)
    got = Encoder(game).legal(obs).numpy()
    np.testing.assert_array_equal(got, np.array(legal))
    np.testing.assert_array_equal(got, legal_mask_from_obs(obs, game))
    assert not got.all() and got[:, 1].all()


def test_encoder_refuses_stacks_the_observation_cannot_carry():
    from headsup.alphaholdem.encoding import Encoder

    with pytest.raises(ValueError, match="stack"):
        Encoder(GameConfig(stack_size=20_000, small_blind=50, big_blind=100, bet_sizes=(0.5, 1.0)))


# ---------------------------------------------------------------------------------------------- network
def test_network_shapes_masking_and_separate_towers(tmp_path):
    from headsup.alphaholdem.model import AlphaNet, load_alpha_net

    torch.manual_seed(0)
    net = AlphaNet(DEFAULT_GAME)
    cards, acts = torch.rand(5, 6, 4, 13) > 0.8, torch.rand(5, 24, 4, 4) > 0.8
    legal = torch.tensor([[False, True, True, True]] * 4 + [[True, True, False, False]])
    logits, value = net(cards, acts, legal)
    assert logits.shape == (5, 4) and value.shape == (5,)
    p = torch.softmax(logits, dim=-1)
    assert (p[~legal] == 0).all() and torch.allclose(p.sum(1), torch.ones(5)) and torch.isfinite(logits).all()
    raw, _ = net(cards, acts)  # without a mask: the raw logits
    assert torch.equal(raw[legal], logits[legal]) and (raw[~legal] > -100).all()
    # pseudo-siamese: one ConvNet per tensor, no shared parameters, both reach the heads
    card_params, action_params = set(map(id, net.card_tower.parameters())), set(map(id, net.action_tower.parameters()))
    assert card_params and action_params and card_params.isdisjoint(action_params)
    assert [m.in_channels for m in net.card_tower if isinstance(m, torch.nn.Conv2d)] == [6, 64, 64]
    assert [m.in_channels for m in net.action_tower if isinstance(m, torch.nn.Conv2d)] == [24, 64, 64]
    assert not torch.equal(net(cards, torch.zeros_like(acts), legal)[0], logits)
    assert not torch.equal(net(torch.zeros_like(cards), acts, legal)[1], value)
    counts = net.parameter_counts()
    assert counts["total"] == sum(p.numel() for p in net.parameters()) == counts["conv"] + counts["fc"]
    assert 1.3e6 < counts["total"] < 1.5e6 and counts["conv"] == 2 * 2 * (64 * 64 * 9 + 64) + (6 + 24) * 64 * 9 + 2 * 64
    # the artefact carries its config incl. the action tree
    path = tmp_path / "policy.pth"
    net.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["config"]["game"] == DEFAULT_GAME.tree_dict() and state["config"]["kind"] == "alphaholdem"
    again = load_alpha_net(path)
    assert not again.training and again.game.tree_dict() == DEFAULT_GAME.tree_dict()
    assert torch.equal(again(cards, acts, legal)[0], net.eval()(cards, acts, legal)[0])
    # a larger tree only changes tensor sizes
    wide = AlphaNet(TREES["five sizes"], channels=8, conv_layers=2, hidden=16)
    logits, value = wide(torch.zeros(3, 6, 4, 13), torch.zeros(3, 24, 4, 8), torch.ones(3, 8, dtype=torch.bool))
    assert logits.shape == (3, 8) and value.shape == (3,)
    assert wide.config == dict(kind="alphaholdem", game=TREES["five sizes"].tree_dict(), channels=8, conv_layers=2, hidden=16)


# ---------------------------------------------------------------------------------------------- Trinal-Clip PPO
def test_trinal_clip_policy_objective_by_hand():
    from headsup.alphaholdem.ppo import trinal_clip_objective

    eps, delta1 = 0.2, 3.0
    ratio = torch.tensor([0.5, 1.0, 1.1, 2.0, 5.0, 0.5, 1.0, 1.1, 2.0, 5.0], requires_grad=True)
    adv = torch.tensor([1.0] * 5 + [-1.0] * 5)
    out = trinal_clip_objective(ratio, adv, eps, delta1)
    # A > 0: PPO, min(r, 1 + eps) A.  A < 0: clamp(r, 1 - eps, delta1) A - the paper's eq. (3)
    np.testing.assert_allclose(out.detach().numpy(), [0.5, 1.0, 1.1, 1.2, 1.2, -0.8, -1.0, -1.1, -2.0, -3.0], rtol=1e-6)
    out.sum().backward()
    # no gradient where a clip is active: r > 1 + eps with A > 0; r < 1 - eps or r > delta1 with A < 0
    np.testing.assert_allclose(ratio.grad.numpy(), [1, 1, 1, 0, 0, 0, -1, -1, -1, 0])
    with torch.no_grad():
        ppo = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
        assert torch.equal(out[:9], ppo[:9]) and out[9] == -3.0 and ppo[9] == -5.0  # the third clip is not a no-op
        eq3 = ratio.clamp(1 - eps, delta1) * adv
        assert torch.allclose(out[5:], eq3[5:])  # for A < 0 it is eq. (3) as printed
        # one re-implementation takes the min over three terms, which is PPO again (the delta1 term never binds)
        third = torch.clamp(ratio, min=ratio.clamp(1 - eps, 1 + eps), max=torch.full_like(ratio, delta1)) * adv
        min3 = torch.minimum(ppo, third)
        assert torch.equal(min3, ppo) and not torch.equal(min3, out)
    # delta1 only matters for A < 0
    pos = trinal_clip_objective(torch.tensor([5.0]), torch.tensor([2.0]), eps, delta1)
    assert pos.item() == pytest.approx(2.4)


def test_value_target_is_clipped_to_the_chips_put_in_so_far():
    from headsup.alphaholdem.encoding import Encoder
    from headsup.alphaholdem.ppo import clipped_value_target

    ret = torch.tensor([0.9, -0.9, 0.03, -0.01, 0.5, -1.0])  # returns in stacks (reward scale 100)
    own = torch.tensor([10.0, 10.0, 2.0, 2.0, 100.0, 100.0])  # chips the player has put in at the state
    opp = torch.tensor([20.0, 20.0, 4.0, 4.0, 100.0, 100.0])  # ... and the opponent
    target = clipped_value_target(ret, own, opp, 100.0)
    np.testing.assert_allclose(target.numpy(), [0.2, -0.1, 0.03, -0.01, 0.5, -1.0], rtol=1e-6)  # [-delta2, delta3] = [-own, opp]
    assert torch.equal(clipped_value_target(ret, own, opp, 100.0, clip=False), ret)
    # the bounds of real states: the small blind's first decision (1 and 2 chips in), then the big blind facing a raise
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    first = e.reset()
    e.step(2)
    facing = e.observation()
    own, opp = Encoder(DEFAULT_GAME).value_bounds(np.stack([first, facing]))
    assert own.tolist() == [1.0, 2.0] and opp.tolist() == [2.0, 4.0]
    target = clipped_value_target(torch.tensor([1.0, -1.0]), own, opp, 100.0)  # an all-in won / lost later in the hand
    np.testing.assert_allclose(target.numpy(), [0.02, -0.02], rtol=1e-6)


def test_per_state_value_clip_biases_the_advantages_of_earlier_decisions():
    """Why the per-state clip is not the default.  One hand at two tables: seat 0 decides twice and then wins / loses
    50 chips.  At its second decision it faces a bet (2 chips in against 10), so the clipped targets are +0.10 and
    -0.02: the critic that fits them says +0.04 where the expected return is 0.  With lambda < 1 the advantage of
    the first decision bootstraps on that value and exceeds the Monte-Carlo advantage by (1 - lambda) x 0.04 on
    average; with the unclipped target (an unbiased critic) the two agree in expectation."""
    from headsup.alphaholdem.ppo import clipped_value_target, stream_gae

    scale, lam = 100.0, 0.95
    seats = torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1]])  # [step, table]: seat 0, seat 1, seat 0, seat 1 ends the hand
    active = torch.ones(4, 2, dtype=torch.bool)
    dones = torch.tensor([[0, 0], [0, 0], [0, 0], [1, 1]]).bool()
    rewards = torch.zeros(4, 2, 2)
    rewards[3, 0] = torch.tensor([50.0, -50.0]) / scale
    rewards[3, 1] = torch.tensor([-50.0, 50.0]) / scale
    returns = torch.tensor([0.5, -0.5])  # seat 0's return at the two tables (gamma 1): mean 0
    chips = {0: (1.0, 2.0), 2: (2.0, 10.0)}  # (own, opponent's) chips in at seat 0's first / second decision
    results = {}
    for clip in (True, False):
        values = torch.zeros(4, 2)
        for t, (own, opp) in chips.items():  # the critic that minimises the value loss: the state's mean target
            values[t] = clipped_value_target(returns, torch.full((2,), own), torch.full((2,), opp), scale, clip=clip).mean()
        gae, _ = stream_gae(values, rewards, dones, seats, active, gamma=1.0, lam=lam)
        monte_carlo, _ = stream_gae(values, rewards, dones, seats, active, gamma=1.0, lam=1.0)
        results[clip] = (values[2, 0].item(), (gae[0] - monte_carlo[0]).mean().item(), (gae[2] - monte_carlo[2]).abs().max().item())
    value, bias, last = results[True]
    assert value == pytest.approx(0.04) and bias == pytest.approx((1 - lam) * 0.04, rel=1e-3) and last == 0.0
    value, bias, last = results[False]
    assert value == pytest.approx(0.0, abs=1e-7) and bias == pytest.approx(0.0, abs=1e-7) and last == 0.0


def test_value_fit_statistics():
    """What the log says about the value head: variance explained of the return and - with the clip on - of its own
    target, and the mean of V - return in chips."""
    from headsup.alphaholdem.ppo import value_fit

    ret = torch.tensor([0.9, -0.9, 0.03, -0.01, 0.5, -1.0])
    value = torch.tensor([0.1, -0.05, 0.02, 0.0, 0.3, -0.2])
    own = torch.tensor([10.0, 10.0, 2.0, 2.0, 100.0, 100.0])
    opp = torch.tensor([20.0, 20.0, 4.0, 4.0, 100.0, 100.0])
    r, v = ret.double().numpy(), value.double().numpy()
    target = np.array([0.2, -0.1, 0.03, -0.01, 0.5, -1.0])  # the returns clipped to [-own, opp] / 100
    plain = value_fit(ret, value, own, opp, 100.0, value_clip=False)
    assert set(plain) == {"explained_variance", "value_bias"}
    assert plain["explained_variance"] == pytest.approx(1 - np.var(r - v, ddof=1) / np.var(r, ddof=1), rel=1e-5)
    assert plain["value_bias"] == pytest.approx(100.0 * (v - r).mean(), rel=1e-5)  # chips
    clipped = value_fit(ret, value, own, opp, 100.0, value_clip=True)
    assert set(clipped) == {"explained_variance", "explained_variance_target", "value_bias"}
    assert clipped["explained_variance_target"] == pytest.approx(1 - np.var(target - v, ddof=1) / np.var(target, ddof=1), rel=1e-5)
    assert clipped["explained_variance"] == plain["explained_variance"] and clipped["value_bias"] == plain["value_bias"]
    assert clipped["explained_variance_target"] > clipped["explained_variance"] + 0.1  # the head fits its target better than the return


def _toy_trajectory():
    """Two tables, eight lock-step steps (gamma 0.9, lambda 0.5).  Table 0, the agent against itself: a four-decision
    hand, a hand that seat 0 open-folds (seat 1 gets +1 without a decision - the two-seat form of the one-seat envs'
    terminal observation after a reset), a two-decision hand, then waiting.  Table 1, against a pool member (its
    cells have value 0): the main agent in seat 0 with a waiting step inside the hand, then a hand the opponent
    open-folds."""
    seats = torch.tensor([[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0]]).T
    active = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 0, 0, 0]]).T.bool()
    dones = torch.tensor([[0, 0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0]]).T.bool()
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 9.0], [1.0, 9.0, 0.0, 2.0, 0.0, 9.0, 9.0, 9.0]]).T
    rewards = torch.zeros(8, 2, 2)
    rewards[3, 0] = torch.tensor([2.0, -2.0])
    rewards[4, 0] = torch.tensor([-1.0, 1.0])
    rewards[6, 0] = torch.tensor([3.0, -3.0])
    rewards[3, 1] = torch.tensor([-4.0, 4.0])
    rewards[4, 1] = torch.tensor([-1.0, 1.0])
    learner = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 1, 0, 0, 0, 0]]).T.bool()
    return values, rewards, dones, seats, active, learner


def test_gae_along_seat_streams_by_hand():
    from headsup.alphaholdem.ppo import stream_gae

    values, rewards, dones, seats, active, learner = _toy_trajectory()
    adv, ret = stream_gae(values, rewards, dones, seats, active, gamma=0.9, lam=0.5)
    # table 0, hand 1: seat 0 decides at t0, t2 and wins 2; seat 1 at t1, t3 and loses 2
    #   A(t2) = 2 - 0.3;  A(t0) = 0.9 * 0.3 - 0.1 + 0.45 * 1.7;  A(t3) = -2 - 0.4;  A(t1) = 0.9 * 0.4 - 0.2 + 0.45 * -2.4
    # hand 2: seat 0 folds at t4 (-1); seat 1's +1 belongs to no decision.  hand 3: t5 (seat 0, +3), t6 (seat 1, -3)
    expected_adv0 = [0.935, -0.92, 1.7, -2.4, -1.5, 2.4, -3.7]
    expected_ret0 = [1.8, -1.8, 2.0, -2.0, -1.0, 3.0, -3.0]
    np.testing.assert_allclose(adv[:7, 0].numpy(), expected_adv0, rtol=1e-5)
    np.testing.assert_allclose(ret[:7, 0].numpy(), expected_ret0, rtol=1e-5)
    # table 1: the main agent's decisions t0, t3 (it loses 4), a waiting step and the opponent's decision in between
    #   A(t3) = -4 - 2;  A(t0) = 0.9 * 2 - 1 + 0.45 * -6
    np.testing.assert_allclose(adv[[0, 3], 1].numpy(), [-1.9, -6.0], rtol=1e-5)
    np.testing.assert_allclose(ret[[0, 3], 1].numpy(), [-3.6, -4.0], rtol=1e-5)
    assert not adv[~active].any() and not ret[~active].any()  # waiting cells carry nothing
    # lambda = 1: the advantage is the return minus the value; gamma = lambda = 1: the hand's reward minus the value
    adv1, ret1 = stream_gae(values, rewards, dones, seats, active, gamma=0.9, lam=1.0)
    np.testing.assert_allclose(adv1[learner].numpy(), (ret1 - values)[learner].numpy(), rtol=1e-5)
    _, ret11 = stream_gae(values, rewards, dones, seats, active, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret11[:7, 0].numpy(), [2, -2, 2, -2, -1, 3, -3])


def test_ppo_loss_puts_the_pieces_together():
    from headsup.alphaholdem.model import AlphaNet
    from headsup.alphaholdem.ppo import clipped_value_target, ppo_loss, trinal_clip_objective

    torch.manual_seed(1)
    net = AlphaNet(DEFAULT_GAME, channels=8, conv_layers=1, hidden=16)
    n = 6
    batch = dict(cards=torch.rand(n, 6, 4, 13) > 0.8, acts=torch.rand(n, 24, 4, 4) > 0.8,
                 legal=torch.tensor([[False, True, True, True]] * 3 + [[True, True, True, True]] * 3),
                 action=torch.tensor([1, 2, 3, 0, 1, 2]), logp=torch.log(torch.tensor([0.3, 0.05, 0.9, 0.25, 0.01, 0.5])),
                 adv=torch.tensor([1.0, -2.0, 0.5, -1.0, -3.0, 2.0]), ret=torch.tensor([0.9, -0.9, 0.01, 0.5, -1.0, 0.0]),
                 own=torch.tensor([2.0, 10.0, 4.0, 1.0, 50.0, 2.0]), opp=torch.tensor([2.0, 20.0, 4.0, 2.0, 50.0, 2.0]))
    loss, stats = ppo_loss(net, batch, eps=0.2, delta1=3.0, value_coef=0.5, entropy_coef=0.01, reward_scale=100.0, value_clip=True)
    with torch.no_grad():
        logits, value = net(batch["cards"], batch["acts"], batch["legal"])
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, batch["action"][:, None]).squeeze(1)
        ratio = torch.exp(logp - batch["logp"])
        policy = trinal_clip_objective(ratio, batch["adv"], 0.2, 3.0).mean()
        target = clipped_value_target(batch["ret"], batch["own"], batch["opp"], 100.0)
        value_loss = ((target - value) ** 2).mean()
        p = logp_all.exp()
        entropy = -(p * logp_all).sum(1).mean()
    assert loss.item() == pytest.approx((-policy + 0.5 * value_loss - 0.01 * entropy).item(), rel=1e-5)
    assert stats["policy"] == pytest.approx(policy.item(), rel=1e-5) and stats["value"] == pytest.approx(value_loss.item(), rel=1e-5)
    assert stats["entropy"] == pytest.approx(entropy.item(), rel=1e-5) and 0 < entropy.item() < np.log(4)
    assert (ratio > 3.0).any() and stats["delta1_clipped"] > 0  # the batch exercises the third clip ...
    assert (target != batch["ret"]).sum() == 4 and stats["value_clipped"] == pytest.approx(4 / 6)  # ... and the value clip
    unclipped, _ = ppo_loss(net, batch, eps=0.2, delta1=3.0, value_coef=0.5, entropy_coef=0.01, reward_scale=100.0, value_clip=False)
    assert unclipped.item() != pytest.approx(loss.item(), rel=1e-3)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


# ---------------------------------------------------------------------------------------------- K-Best pool
def test_elo_update_and_k_best_selection():
    from headsup.alphaholdem.model import AlphaNet
    from headsup.alphaholdem.pool import KBestPool, elo_expected

    assert elo_expected(1200.0, 1200.0) == 0.5
    assert elo_expected(1600.0, 1200.0) == pytest.approx(10 / 11) and elo_expected(1200.0, 1600.0) == pytest.approx(1 / 11)
    torch.manual_seed(0)
    net = AlphaNet(DEFAULT_GAME, channels=4, conv_layers=1, hidden=8)
    pool = KBestPool(k=2, k_factor=16.0, initial_elo=1200.0)
    assert len(pool) == 0 and pool.main_elo == 1200.0
    assert pool.add(net, iteration=10) is None and pool.members[0].elo == 1200.0 and pool.members[0].iteration == 10
    pool.record(0, chips=35.0, hands=50)  # the main agent wins the block of hands: one game, score 1
    assert pool.main_elo == pytest.approx(1208.0) and pool.members[0].elo == pytest.approx(1192.0)
    expected = 1.0 / (1.0 + 10 ** ((1192.0 - 1208.0) / 400.0))
    pool.record(0, chips=-0.5, hands=50)  # lost: score 0
    assert pool.main_elo == pytest.approx(1208.0 - 16.0 * expected) and pool.members[0].elo == pytest.approx(1192.0 + 16.0 * expected)
    before = (pool.main_elo, pool.members[0].elo)
    pool.record(0, chips=0.0, hands=0)  # no hands: no game
    assert (pool.main_elo, pool.members[0].elo) == before
    pool.main_elo, pool.members[0].elo = 1220.0, 1180.0
    pool.record(0, chips=0.0, hands=10)  # a draw moves the ratings towards each other
    assert 1200.0 < pool.main_elo < 1220.0 and 1180.0 < pool.members[0].elo < 1200.0
    assert pool.main_elo + pool.members[0].elo == pytest.approx(2400.0)  # zero-sum
    # K best survive: a snapshot enters with the main agent's rating, the lowest rating leaves
    pool.main_elo, pool.members[0].elo = 1300.0, 1250.0
    assert pool.add(net, iteration=20) is None and [m.elo for m in pool.members] == [1250.0, 1300.0]
    pool.main_elo = 1280.0
    assert pool.add(net, iteration=30) == 10  # the member of iteration 10 (1250) is the weakest of the three
    assert [(m.iteration, m.elo) for m in pool.members] == [(20, 1300.0), (30, 1280.0)]
    pool.main_elo = 1100.0
    assert pool.add(net, iteration=40) == 40 and [m.iteration for m in pool.members] == [20, 30]  # too weak to enter
    # members are frozen copies
    member = pool.members[0].net
    assert member is not net and not member.training and not any(p.requires_grad for p in member.parameters())
    with torch.no_grad():
        next(net.parameters()).add_(1.0)
    assert not torch.equal(next(member.parameters()), next(net.parameters()))


def test_pool_assigns_opponents_evenly_and_round_trips():
    from headsup.alphaholdem.model import AlphaNet
    from headsup.alphaholdem.pool import KBestPool

    net = AlphaNet(DEFAULT_GAME, channels=4, conv_layers=1, hidden=8)
    pool = KBestPool(k=3)
    rng = np.random.default_rng(0)
    assert (pool.assign(10, rng) == -1).all()  # an empty pool: the agent plays itself
    for it in (5, 10, 15):
        pool.add(net, iteration=it)
    a = pool.assign(4096, rng)
    assert sorted(np.unique(a).tolist()) == [-1, 0, 1, 2]  # the current agent and every survivor
    assert np.bincount(a + 1).tolist() == [1024] * 4 and not (a == np.sort(a)).all()  # uniform, in random order
    assert sorted(np.bincount(pool.assign(10, rng) + 1, minlength=4).tolist()) == [2, 2, 3, 3]
    pool.members[1].elo, pool.main_elo = 1234.5, 1111.0
    other = KBestPool(k=3)
    other.load_state_dict(pool.state_dict())
    assert other.main_elo == 1111.0 and [(m.iteration, m.elo) for m in other.members] == [(5, 1200.0), (10, 1234.5), (15, 1200.0)]
    for m, o in zip(pool.members, other.members):
        assert all(torch.equal(x, y) for x, y in zip(m.net.parameters(), o.net.parameters()))
        assert not any(p.requires_grad for p in o.net.parameters())


# ---------------------------------------------------------------------------------------------- trainer
TINY = dict(envs=48, samples=500, epochs=2, minibatch=200, channels=8, conv_layers=1, hidden=32, pool=2, snapshot_every=1, seed=0)


def _force(net, action):
    """Make a network (nearly) deterministic: ``action`` when it is legal, else check / call."""
    with torch.no_grad():
        net.policy_head.weight.zero_()
        net.policy_head.bias.copy_(torch.tensor([60.0 if a == action else 30.0 if a == 1 else 0.0 for a in range(net.num_actions)]))


def test_rollout_trains_on_the_main_agents_decisions_only():
    """The main agent always raises, the pool member folds whenever it faces a bet.  As small blind the main agent
    raises and wins the big blind (+2): one sample per hand, action 2, return 2 / scale.  As big blind it wins the
    small blind without a decision (+1): no sample, but the hand counts in the result against the member.  Any
    sample with another action or return would be one of the opponent's decisions or a misplaced reward."""
    from headsup.alphaholdem.train import Trainer

    trainer = Trainer(TINY, device="cpu")
    trainer.pool.add(trainer.net, iteration=0)
    _force(trainer.pool.members[0].net, 0)
    _force(trainer.net, 2)
    batch, info = trainer.collect(assignment=np.zeros(48, dtype=np.int64))
    n = len(batch["action"])
    assert n >= 500 and all(len(v) == n for v in batch.values())
    assert (batch["action"] == 2).all()
    np.testing.assert_allclose(batch["ret"].numpy(), 0.02, rtol=1e-6)  # +2 chips in stacks of 100, on the deciding sample
    assert (batch["own"] == 1).all() and (batch["opp"] == 2).all() and (batch["seat"] == 0).all()
    hands, chips = int(info["pool_hands"][0]), float(info["pool_chips"][0])
    assert info["hands"] == hands and n == round(chips) - hands  # chips = 2 a + b and hands = a + b: a hands as small blind
    assert abs(n / hands - 0.5) < 0.05  # the seat alternates from hand to hand (fixed seats would give 1/3: those hands are longer)
    np.testing.assert_allclose(batch["adv"].numpy(), (batch["ret"] - batch["value"]).numpy(), rtol=1e-4, atol=1e-7)


def test_rollout_returns_are_each_seats_reward_of_its_hand():
    """Self-play (an empty pool): every decision of both seats is a sample; the rollout ends at hand boundaries; a
    sample's return is its seat's reward of that hand, discounted per later decision of the seat - recomputed here
    with a forward walk over the recorded grid.  The stored inputs, actions, log-probabilities and values belong
    together (the network reproduces them)."""
    from headsup.alphaholdem.train import Trainer

    from headsup.alphaholdem.ppo import stream_gae

    trainer = Trainer({**TINY, "gamma": 0.9, "lam": 0.3, "allin_ev": True}, device="cpu")
    batch, info = trainer.collect()
    grid = info["grid"]
    for lam, same in ((0.3, True), (0.95, False)):  # the advantages are GAE with the trainer's gamma and lambda
        adv, _ = stream_gae(grid["value"], grid["reward"] / 100.0, grid["done"], grid["seat"], grid["active"], 0.9, lam)
        assert torch.allclose(adv[batch["step"], batch["table"]], batch["adv"], atol=1e-6) == same
    seat, active, done, reward = (grid[k].numpy() for k in ("seat", "active", "done", "reward"))
    T, N = seat.shape
    assert len(batch["action"]) == active.sum() >= 500 and info["hands"] == (done & active).sum()
    expected = {}
    for i in range(N):
        steps = np.flatnonzero(active[:, i])
        assert done[steps[-1], i]  # no hand is cut off
        hand = {0: [], 1: []}
        for t in steps:
            hand[seat[t, i]].append(t)
            if done[t, i]:
                for s in (0, 1):
                    for j, tt in enumerate(hand[s]):
                        expected[(tt, i)] = 0.9 ** (len(hand[s]) - 1 - j) * reward[t, i, s] / 100.0
                hand = {0: [], 1: []}
    got = {(int(t), int(i)): float(r) for t, i, r in zip(batch["step"], batch["table"], batch["ret"])}
    assert got.keys() == expected.keys()
    np.testing.assert_allclose([got[k] for k in expected], list(expected.values()), rtol=1e-5, atol=1e-7)
    assert (reward != np.rint(reward)).any() and len({round(abs(v), 6) for v in expected.values()}) > 5  # all-in EV; many pot sizes
    np.testing.assert_array_equal(batch["seat"].numpy(), seat[batch["step"].numpy(), batch["table"].numpy()])
    with torch.no_grad():
        logits, value = trainer.net(batch["cards"], batch["acts"], batch["legal"])
    logp = torch.log_softmax(logits, dim=-1).gather(1, batch["action"][:, None]).squeeze(1)
    assert torch.allclose(logp, batch["logp"], atol=1e-5) and torch.allclose(value, batch["value"], atol=1e-5)
    assert batch["legal"].gather(1, batch["action"][:, None]).all()
    assert batch["cards"][:, 0].sum(dim=(1, 2)).eq(2).all()  # two hole cards in every sample


def test_rollout_samples_are_aligned_with_their_observations():
    """Half of the tables against a pool member, half self-play.  Which cells of the [step, table] grid are samples
    is replayed here from the assignment and the seats (the main agent's seat flips when a hand against the member
    ends); every sample's inputs, legal set and value-clip bounds are those of the observation at its cell."""
    from headsup.alphaholdem.encoding import Encoder
    from headsup.alphaholdem.train import Trainer

    trainer = Trainer(TINY, device="cpu")
    trainer.pool.add(trainer.net, iteration=0)
    batch, info = trainer.collect(keep_obs=True)
    grid = info["grid"]
    seat, active, done = (grid[k].numpy() for k in ("seat", "active", "done"))
    opponent, main_seat = info["opponent"], info["main_seat"].copy()
    assert sorted(np.bincount(opponent + 1).tolist()) == [24, 24]
    expected = np.zeros_like(active)
    for t in range(len(seat)):
        expected[t] = active[t] & ((opponent < 0) | (seat[t] == main_seat))
        main_seat[done[t] & (opponent >= 0)] ^= 1
    cells = np.zeros_like(active)
    cells[batch["step"].numpy(), batch["table"].numpy()] = True
    np.testing.assert_array_equal(cells, expected)
    assert cells.sum() == len(batch["action"]) and 0.6 < cells.sum() / active.sum() < 0.9  # all of self-play, half of the rest
    obs = grid["obs"][batch["step"], batch["table"]]
    enc = Encoder(DEFAULT_GAME)
    cards, acts, legal = enc(obs)
    assert torch.equal(cards, batch["cards"]) and torch.equal(acts, batch["acts"]) and torch.equal(legal, batch["legal"])
    own, opp = enc.value_bounds(obs)
    assert torch.equal(own, batch["own"]) and torch.equal(opp, batch["opp"]) and len(set(own.tolist())) > 5
    assert torch.equal(obs[:, 22].long(), batch["seat"])  # the observer is the acting seat
    assert torch.equal(grid["value"][batch["step"], batch["table"]], batch["value"]) and not grid["value"].numpy()[~cells].any()


def test_elo_game_follows_the_rollout_result():
    """The main agent always raises and the pool member folds to every bet: the main agent wins every hand against it
    (+2 as small blind, +1 as big blind), so its rating rises by K / 2 and the member's falls."""
    from headsup.alphaholdem.train import Trainer

    trainer = Trainer(TINY, device="cpu", snapshot_every=100)
    trainer.pool.add(trainer.net, iteration=0)
    _force(trainer.pool.members[0].net, 0)
    _force(trainer.net, 2)
    record = trainer.iterate()
    assert record["hands_vs_pool"] > 50 and record["chips_vs_pool"] == pytest.approx(1.5, abs=0.1)
    assert trainer.pool.main_elo == pytest.approx(1208.0) and trainer.pool.members[0].elo == pytest.approx(1192.0)
    assert record["elo"] == trainer.pool.main_elo and record["pool"] == [[0, trainer.pool.members[0].elo]]


def test_update_uses_the_configured_loss():
    """One minibatch, one epoch: the loss of the update is the Trinal-Clip loss with the trainer's settings on the
    rollout with normalised advantages (distinct values for every coefficient: a swapped pair would show)."""
    import copy

    from headsup.alphaholdem.ppo import ppo_loss
    from headsup.alphaholdem.train import Trainer

    settings = dict(eps=0.1, delta1=1.5, value_coef=0.7, entropy_coef=0.03, reward_scale=50.0, epochs=1, minibatch=10**6, value_clip=True)
    trainer = Trainer({**TINY, **settings}, device="cpu")
    batch, _ = trainer.collect()
    batch["logp"] = batch["logp"] + torch.linspace(-1.5, 1.5, len(batch["logp"]))  # ratios from 0.2 to 4.5: every clip is active
    reference = copy.deepcopy(trainer.net).train()
    before = [p.detach().clone() for p in trainer.net.parameters()]
    stats = trainer.update(batch)
    normalised = dict(batch, adv=(batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8))
    loss, expected = ppo_loss(reference, normalised, eps=0.1, delta1=1.5, value_coef=0.7, entropy_coef=0.03, reward_scale=50.0,
                              value_clip=True)
    assert stats["loss"] == pytest.approx(loss.item(), rel=1e-4)
    for k in ("policy", "value", "entropy", "clipped", "delta1_clipped", "value_clipped"):
        assert stats[k] == pytest.approx(expected[k], rel=1e-4), k
    assert expected["clipped"] > 0.5 and expected["delta1_clipped"] > 0.05 and 0.2 < expected["value_clipped"] < 1.0
    assert loss.item() != pytest.approx(ppo_loss(reference, normalised, eps=0.1, delta1=1.5, value_coef=0.03, entropy_coef=0.7,
                                                 reward_scale=50.0, value_clip=True)[0].item(), rel=1e-3)
    moved = [float((a - b.detach()).abs().max()) for a, b in zip(before, trainer.net.parameters())]
    assert 0 < max(moved) < 1e-3 and stats["grad_norm"] > 0  # one Adam step of lr 3e-4
    plain = Trainer({**TINY, **settings, "adv_norm": False, "value_clip": False}, device="cpu")  # the two switches
    plain.net.load_state_dict(reference.state_dict())
    stats = plain.update(batch)
    loss, _ = ppo_loss(reference, batch, eps=0.1, delta1=1.5, value_coef=0.7, entropy_coef=0.03, reward_scale=50.0, value_clip=False)
    assert stats["loss"] == pytest.approx(loss.item(), rel=1e-4) and stats["value_clipped"] == 0


def test_value_clip_is_off_by_default_and_a_flag_pair(tmp_path, monkeypatch):
    """Rollouts hold complete hands, so under the per-hand reading of delta2 / delta3 the paper's value target never
    clips: no clip is the default.  --value-clip turns the per-state reading on, --no-value-clip is still accepted,
    and a resumed run keeps what its checkpoint stored."""
    import inspect

    from headsup.alphaholdem.ppo import ppo_loss
    from headsup.alphaholdem.train import DEFAULTS, Trainer, build_parser

    assert DEFAULTS["value_clip"] is False and Trainer(TINY, device="cpu").cfg["value_clip"] is False
    assert inspect.signature(ppo_loss).parameters["value_clip"].default is False
    parse = lambda *flags: build_parser().parse_args(["--out", "x", "--iterations", "1", *flags])
    assert parse().value_clip is False and parse("--value-clip").value_clip is True
    assert parse("--no-value-clip").value_clip is False and parse("--value-clip", "--no-value-clip").value_clip is False
    Trainer({**TINY, "value_clip": True}, device="cpu").save(tmp_path)
    assert Trainer.resume(tmp_path, device="cpu").cfg["value_clip"] is True
    monkeypatch.setitem(DEFAULTS, "value_clip", True)  # the flags' default is the settings' default, whatever it is
    assert parse().value_clip is True and parse("--no-value-clip").value_clip is False


def test_iteration_logs_the_value_heads_fit():
    from headsup.alphaholdem.ppo import value_fit
    from headsup.alphaholdem.train import Trainer

    for clip, keys in ((False, ("explained_variance", "value_bias")), (True, ("explained_variance", "explained_variance_target", "value_bias"))):
        trainer = Trainer({**TINY, "value_clip": clip}, device="cpu")
        rollouts, collect = [], trainer.collect
        trainer.collect = lambda: rollouts.append(collect()) or rollouts[-1]
        record = trainer.iterate()
        batch = rollouts[0][0]
        expected = value_fit(batch["ret"], batch["value"], batch["own"], batch["opp"], trainer.reward_scale, clip)
        assert set(expected) == set(keys) and all(record[k] == expected[k] and np.isfinite(record[k]) for k in keys)
        assert ("explained_variance_target" in record) == clip


def test_each_pool_member_plays_its_tables_and_gets_its_result():
    """Two pool members with different forced behaviour against a main agent that always raises: member 0 folds to
    every bet (the main agent wins 2 as small blind and 1 as big blind), member 1 always calls (every hand reaches
    the showdown with 10 chips each in the pot).  The stakes at a table show which member played it; results and ELO
    games are per member."""
    from headsup.alphaholdem.pool import elo_expected
    from headsup.alphaholdem.train import Trainer

    trainer = Trainer({**TINY, "envs": 60, "pool": 3, "snapshot_every": 100}, device="cpu")
    for action in (0, 1):
        trainer.pool.add(trainer.net, iteration=action)
        _force(trainer.pool.members[action].net, action)
    _force(trainer.net, 2)
    batch, info = trainer.collect()
    opponent, grid = info["opponent"], info["grid"]
    done, reward = grid["done"].numpy(), grid["reward"].numpy()
    assert np.bincount(opponent + 1).tolist() == [20, 20, 20]
    stakes = [set(np.abs(reward[:, opponent == k, 0][done[:, opponent == k]]).tolist()) for k in (0, 1)]
    assert stakes[0] == {1.0, 2.0} and stakes[1] <= {0.0, 10.0} and 10.0 in stakes[1]  # each table: its member's way to play
    main_seat, chips, hands = info["main_seat"].copy(), np.zeros(2), np.zeros(2, dtype=np.int64)
    for t in range(len(done)):  # the main agent's result against each member, replayed from the grid
        for i in np.flatnonzero(done[t] & (opponent >= 0)):
            chips[opponent[i]] += reward[t, i, main_seat[i]]
            hands[opponent[i]] += 1
            main_seat[i] ^= 1
    np.testing.assert_array_equal(info["pool_hands"], hands)
    np.testing.assert_allclose(info["pool_chips"], chips)
    assert hands.min() >= 20 and chips[0] / hands[0] == pytest.approx(1.5, abs=0.1)
    record = trainer.iterate()  # another rollout, the update and the two ELO games
    (it0, mean0, hands0), (it1, mean1, hands1) = record["vs_pool"]  # per member: its iteration, chips / hand, hands
    assert (it0, it1) == (0, 1) and min(hands0, hands1) >= 20 and mean0 == pytest.approx(1.5, abs=0.1)
    assert record["hands_vs_pool"] == hands0 + hands1
    assert record["chips_vs_pool"] == pytest.approx((mean0 * hands0 + mean1 * hands1) / (hands0 + hands1))
    folder, caller = trainer.pool.members
    assert folder.elo == pytest.approx(1192.0)  # it lost its game against the main agent (then 1208)
    score = 1.0 if mean1 > 0 else 0.0 if mean1 < 0 else 0.5  # the caller's game: whatever the cards gave
    delta = 16.0 * (score - elo_expected(1208.0, 1200.0))
    assert caller.elo == pytest.approx(1200.0 - delta) and trainer.pool.main_elo == pytest.approx(1208.0 + delta)


def test_update_clips_the_gradient_norm():
    """Adam moves a weight by lr x g / (|g| + 1e-5) in its first step: about lr for an ordinary gradient, far less
    when the gradient was clipped to a norm far below 1e-5."""
    from headsup.alphaholdem.train import Trainer

    moved = {}
    for norm in (1e-9, 0.5):
        trainer = Trainer({**TINY, "epochs": 1, "minibatch": 10**6, "max_grad_norm": norm}, device="cpu")
        batch, _ = trainer.collect()
        before = [p.detach().clone() for p in trainer.net.parameters()]
        assert trainer.update(batch)["grad_norm"] > 1e-3  # the norm before clipping
        moved[norm] = max(float((a - b.detach()).abs().max()) for a, b in zip(before, trainer.net.parameters()))
    assert moved[1e-9] < 1e-6 and 1e-4 < moved[0.5] < 1e-3


def test_env_and_evaluation_use_all_in_ev_as_configured(monkeypatch):
    import headsup.env as env_module
    from headsup.alphaholdem.train import Trainer

    trainer = Trainer({**TINY, "allin_ev": True, "ev_samples": 7}, device="cpu")
    assert trainer.env.allin_ev is True and trainer.env.ev_samples == 7  # training rewards: as the settings say
    plain = Trainer(TINY, device="cpu")
    assert plain.env.allin_ev is False and plain.env.ev_samples == 100
    seen, play_hands = [], env_module.play_hands

    def spy(env, agent, hands, **kwargs):
        seen.append(kwargs.get("allin_ev"))
        return play_hands(env, agent, hands, **kwargs)

    monkeypatch.setattr(env_module, "play_hands", spy)
    out = plain.evaluate(128, opponents=("allin", "call"), num_envs=64)  # evaluation: always all-in EV, like the tools
    assert seen == [True, True] and set(out) == {"allin", "call"} and all(len(v) == 2 and v[1] > 0 for v in out.values())


def test_training_moves_the_main_agent_and_not_the_pool():
    from headsup.alphaholdem.train import Trainer

    trainer = Trainer(TINY, device="cpu")
    start = [p.detach().clone() for p in trainer.net.parameters()]
    first = trainer.iterate()
    assert trainer.iteration == 1 and len(trainer.pool) == 1 and trainer.pool.members[0].iteration == 1  # snapshot_every = 1
    assert any(not torch.equal(a, b) for a, b in zip(start, trainer.net.parameters()))
    member = trainer.pool.members[0].net
    frozen = [p.detach().clone() for p in member.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(frozen, trainer.net.parameters()))  # the snapshot is the agent after iteration 1
    second = trainer.iterate()
    assert all(torch.equal(a, b) for a, b in zip(frozen, member.parameters()))  # the frozen opponent did not learn
    assert any(not torch.equal(a, b) for a, b in zip(frozen, trainer.net.parameters()))  # the main agent did
    assert not any(p.requires_grad for p in member.parameters()) and all(p.requires_grad for p in trainer.net.parameters())
    rated = trainer.pool.members[0]
    assert rated.net is member and trainer.pool.main_elo != 1200.0 and trainer.pool.main_elo + rated.elo == pytest.approx(2400.0)  # one game
    assert len(trainer.pool) == 2 and second["pool"][0][0] == 1 and second["hands_vs_pool"] > 0 and first["hands_vs_pool"] == 0
    for record in (first, second):
        assert all(np.isfinite(record[k]) for k in ("policy", "value", "entropy", "kl", "elo", "samples_per_second"))
        assert record["batch"] >= 500 and 0.0 < record["entropy"] <= np.log(4) and sum(record["actions"]) == pytest.approx(1.0)
    assert second["samples"] == first["batch"] + second["batch"] == trainer.samples and trainer.log == [first, second]


def test_checkpoint_round_trip_and_cli(tmp_path, capsys):
    import json

    from headsup.alphaholdem.train import Trainer, main

    out = tmp_path / "run"
    args = ["--out", str(out), "--envs", "48", "--samples", "400", "--epochs", "1", "--minibatch", "200", "--channels", "8",
            "--conv-layers", "1", "--hidden", "32", "--pool", "2", "--snapshot-every", "1", "--eval-every", "2", "--eval-hands", "300",
            "--checkpoint-every", "1", "--device", "cpu"]
    main(args + ["--iterations", "2"])
    assert {p.name for p in out.iterdir()} >= {"checkpoint.pt", "policy.pth", "log.json"}
    log = json.loads((out / "log.json").read_text())
    assert [r["iteration"] for r in log["log"]] == [1, 2] and log["config"]["samples"] == 400
    assert set(log["log"][1]["eval"]) == {"random", "call", "allin"} and len(log["log"][1]["eval"]["call"]) == 2  # mean, se
    assert "eval" not in log["log"][0]
    a = Trainer.resume(out, device="cpu")
    b = Trainer.resume(out, device="cpu")
    assert a.iteration == 2 and a.samples == log["log"][1]["samples"] and a.cfg["hidden"] == 32 and len(a.pool) == 2
    saved = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    for k, v in a.net.state_dict().items():
        assert torch.equal(v, saved["net"][k])
    assert [(m.iteration, m.elo) for m in a.pool.members] == [(m["iteration"], m["elo"]) for m in saved["pool"]["members"]]
    assert a.pool.main_elo == saved["pool"]["main_elo"] and a.log == log["log"]
    moments = [s["exp_avg"] for s in a.opt.state_dict()["state"].values()]
    assert len(moments) == len(list(a.net.parameters())) and any(m.abs().sum() > 0 for m in moments)  # Adam's state came along
    assert all(torch.equal(x, y) for x, y in zip(moments, [s["exp_avg"] for s in saved["opt"]["state"].values()]))
    assert a.rng.bit_generator.state == saved["rng"] and torch.equal(a.gen.get_state(), saved["gen"])  # both generators
    assert a.rng.bit_generator.state != np.random.default_rng(0).bit_generator.state
    ra, rb = a.iterate(), b.iterate()  # the same checkpoint continues the same way
    assert ra["batch"] == rb["batch"] and ra["policy"] == pytest.approx(rb["policy"], rel=1e-4) and a.iteration == 3
    first_deal = Trainer(saved["config"], device="cpu").env.reset()[0]
    assert not np.array_equal(Trainer.resume(out, device="cpu").env.reset()[0], first_deal)  # ... with new cards, not the first run's
    capsys.readouterr()
    main(args + ["--iterations", "3", "--resume"])
    assert "ignored" not in capsys.readouterr().out  # the flags repeat what the checkpoint stores: nothing to warn about
    log = json.loads((out / "log.json").read_text())
    assert [r["iteration"] for r in log["log"]] == [1, 2, 3]
    resumed = main(args + ["--iterations", "3", "--resume", "--value-clip", "--lr", "0.5"])  # hyperparameters come from the checkpoint
    warning = [line for line in capsys.readouterr().out.splitlines() if "ignored" in line]
    assert len(warning) == 1 and "--value-clip" in warning[0] and "--lr" in warning[0] and "--envs" not in warning[0]
    assert resumed.cfg["lr"] == 3e-4 and resumed.cfg["value_clip"] is False
    with pytest.raises(SystemExit):
        main(args + ["--iterations", "3"])  # an existing run is not overwritten without --resume


# ---------------------------------------------------------------------------------------------- player
def _decision_observations(hands=120, seed=5):
    return np.stack([d["obs"] for d in _random_hands(DEFAULT_GAME, hands, seed)[0]])


def test_player_probabilities_legality_and_batching(tmp_path):
    from headsup.alphaholdem.model import AlphaNet
    from headsup.alphaholdem.player import AlphaHoldemPlayer
    from headsup.players import make_player

    torch.manual_seed(3)
    net = AlphaNet(DEFAULT_GAME, channels=8, conv_layers=1, hidden=32)
    with torch.no_grad():
        net.policy_head.weight.mul_(100.0)  # a peaked policy, with the fold often on top
        net.policy_head.bias[0] = 5.0
    path = tmp_path / "policy.pth"
    net.save(path)
    obs = _decision_observations()
    legal = legal_mask_from_obs(obs, DEFAULT_GAME)
    player = make_player(f"alpha:{path}", device="cpu", seed=0)
    assert isinstance(player, AlphaHoldemPlayer) and player.game.tree_dict() == DEFAULT_GAME.tree_dict()
    assert not getattr(player, "wants_ids", False)
    probs = player.probs(obs)
    assert probs.shape == (len(obs), 4) and probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs[~legal] == 0).all() and (~legal).any() and (probs >= 0).all()
    # stateless: the same answer for a batch, row by row, in chunks and with ids
    rows = np.concatenate([player.probs(obs[i : i + 1]) for i in range(0, len(obs), 7)])
    np.testing.assert_allclose(rows, probs[::7], atol=1e-6)
    small = AlphaHoldemPlayer(str(path), device="cpu", chunk=50)
    np.testing.assert_allclose(small.probs(obs, ids=np.arange(len(obs))), probs, atol=1e-6)
    # sampled actions are legal and follow the probabilities; deterministic = the most likely action
    counts = np.zeros_like(probs)
    for _ in range(200):
        a = player(obs)
        assert a.shape == (len(obs),) and a.dtype == np.int64 and legal[np.arange(len(obs)), a].all()
        counts[np.arange(len(obs)), a] += 1
    assert np.abs(counts / 200 - probs).max() < 0.2 and player.last_probs.shape == probs.shape
    greedy = make_player(f"alpha:{path}", device="cpu", deterministic=True)
    np.testing.assert_array_equal(greedy(obs), probs.argmax(axis=1))
    # terminal observations (after an opponent's open-fold in a one-seat env) are answered too
    e = HeadsUpPoker(rng=np.random.default_rng(0))
    e.reset()
    e.step(0)
    assert player(e.observation(1)[None]).shape == (1,)
    with pytest.raises(ValueError, match="alpha:<"):
        make_player("alpha")
    from headsup.model import BaseModel

    other = tmp_path / "cfr.pth"
    BaseModel().save(other)
    with pytest.raises(ValueError, match="AlphaHoldem"):
        make_player(f"alpha:{other}", device="cpu")


def test_evaluation_tools_take_the_player(tmp_path):
    from headsup import native
    from headsup.alphaholdem.model import AlphaNet
    from headsup.compare import head_to_head

    torch.manual_seed(0)
    path = tmp_path / "policy.pth"
    AlphaNet(DEFAULT_GAME, channels=8, conv_layers=1, hidden=32).save(path)
    mean, se = head_to_head(f"alpha:{path}", "call", 400, num_envs=64, seed=0, device="cpu")
    assert np.isfinite(mean) and se > 0
    if native.available():
        from headsup.lbr import LocalBestResponse

        lbr = LocalBestResponse(f"alpha:{path}", num_tables=4, device="cpu", seed=0, mc_samples=20, workers=2)
        results = lbr.play(4, progress=False)
        assert len(results) == 4 and np.isfinite(results).all() and lbr.summary(results)["policy"] == f"alpha:{path}"
    from headsup.web.session import bot_label

    assert bot_label("alpha:runs/x/policy.pth") == "AlphaHoldem · policy.pth"
