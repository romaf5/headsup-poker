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
    loss, stats = ppo_loss(net, batch, eps=0.2, delta1=3.0, value_coef=0.5, entropy_coef=0.01, reward_scale=100.0)
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
