import numpy as np
import pytest

from headsup import native
from headsup.cards import hand_strength
from headsup.engine import HeadsUpPoker

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")


def test_eval7_matches_treys():
    cpp = native.module()
    rng = np.random.default_rng(0)
    for _ in range(3000):
        cards = rng.permutation(52)[:7].tolist()
        assert cpp.eval7(cards) == hand_strength(cards[:2], cards[2:])


def test_cpp_engine_matches_python_engine():
    cpp = native.module()
    rng = np.random.default_rng(2)
    py = HeadsUpPoker(rng=rng)
    ce = cpp.Engine()
    for _ in range(3000):
        perm = rng.permutation(52)
        o_py = py.reset(deck=perm)
        ce.reset(perm.tolist())
        while True:
            assert py.current == ce.current
            np.testing.assert_array_equal(o_py, ce.observation())
            a = int(rng.choice(4, p=[0.1, 0.4, 0.35, 0.15]))
            o_py, r_py, d_py, _ = py.step(a)
            d_c = ce.step(a)
            assert d_py == d_c
            if d_py:
                assert list(r_py) == list(ce.rewards) and py.folded == ce.folded
                break


def test_native_vecenv_rewards_are_consistent():
    from headsup.env import NativeVecEnv, PokerVecEnv, play_hands
    from headsup.players import AlwaysCallPlayer, RandomPlayer

    # calling station vs calling station: low variance, so means/stds are tightly comparable
    r_native = play_hands(NativeVecEnv(512, "call", seed=0), AlwaysCallPlayer(), 40000)
    r_py = play_hands(PokerVecEnv(512, AlwaysCallPlayer(), seed=0), AlwaysCallPlayer(), 40000)
    assert abs(r_native.mean() - r_py.mean()) < 0.15
    assert abs(r_native.std() - r_py.std()) < 0.5
    # random opponent: fold is never taken when nothing is to call (both implementations)
    r_native = play_hands(NativeVecEnv(512, "random", seed=0), AlwaysCallPlayer(), 20000)
    r_py = play_hands(PokerVecEnv(512, RandomPlayer(seed=0), seed=0), AlwaysCallPlayer(), 20000)
    assert abs(r_native.mean() - r_py.mean()) < 3 * (r_native.std() + r_py.std()) / np.sqrt(20000)


def test_sampling_never_returns_a_zero_probability_action():
    """float32 partial sums can stop short of 1: the remainder belongs to the last action WITH probability (the
    last index was returned - in a limit game the illegal raise at the cap, ~1e-8 per draw; DREAM then stored
    advantage rows with weight t / 1e-12 and the sampled subgame solver read child -1)."""
    cpp = native.module()
    p = np.array([0.5, 0.49999994, 0.0], dtype=np.float32)
    u = np.float32(0.99999994)
    assert np.float32(p[0] + p[1]) <= u < 1.0  # the draw lies beyond the partial sums
    assert cpp.sample_index(p, float(u)) == 1
    assert cpp.sample_index(np.array([0.0, 1.0, 0.0], np.float32), 0.0) == 1
    assert cpp.sample_index(np.array([0.25, 0.0, 0.75], np.float32), 0.25) == 2
    assert cpp.sample_index(np.array([0.25, 0.0, 0.75], np.float32), 0.2) == 0


def test_native_fixed_bots_play_the_twin_of_a_masked_action():
    """The native raise bot played action 2 even where the game masks it; the Python bot plays its twin - the same
    chips, but another raise counter in what the agent then observes, and replay-based players left their tree."""
    from headsup.env import make_vec_env
    from headsup.game import GameConfig

    game = GameConfig(bet_sizes=(0.5, 1.0), mask_redundant=True)
    seen = {}
    for backend in ("cpp", "python"):
        env = make_vec_env(64, "raise", seed=0, game=game, backend=backend)
        obs, rng, counters = env.reset(), np.random.default_rng(0), set()
        for _ in range(300):
            facing_allin = (obs[:, 28] > 0) & (np.abs(obs[:, 23] - obs[:, 28]) < 1e-6)  # to call == the whole remaining stack
            counters |= set(obs[facing_allin, 79].astype(int).tolist())
            obs, _, _, _ = env.step(np.where(rng.random(64) < 0.5, 2, 1))
        seen[backend] = counters
    assert seen["cpp"] == seen["python"] and seen["cpp"]


def test_vec_env_rejects_out_of_range_actions():
    from headsup.env import make_vec_env

    env = make_vec_env(4, "call", seed=0, backend="cpp")
    env.reset()
    for bad in (7, -1):
        with pytest.raises(ValueError, match="[Ii]nvalid action"):
            env.step(np.array([1, bad, 1, 1]))


def test_solvers_map_masked_actions_to_their_twins_and_bound_the_buckets():
    from headsup.game import GameConfig

    cpp = native.module()
    game = GameConfig(bet_sizes=(0.5, 1.0), mask_redundant=True)
    e = cpp.Engine(native.engine_config(game=game))
    e.reset(list(range(9)))
    while e.stage == 0:  # to the flop by calling
        e.step(1)
    sv = cpp.VectorSolver()
    sv.build(e, 50)
    legal = np.asarray(e.legal_mask())
    assert not legal.all()  # the flop root of this game has a masked action (fold: nothing to call)
    for a in np.flatnonzero(~legal):
        assert sv.child(0, int(a)) >= 0  # executed as its twin, as the engine does
    with pytest.raises((RuntimeError, ValueError), match="buckets"):
        cpp.VectorSolver().build(e, 3000)  # more buckets than hands overflowed the per-level scratch


def test_engine_config_is_validated():
    cpp = native.module()
    cfg = cpp.EngineConfig()
    cfg.bet_sizes = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # 11 actions > MAX_ACTIONS
    with pytest.raises((RuntimeError, ValueError), match="actions"):
        cpp.Engine(cfg)
    cfg = cpp.EngineConfig()
    cfg.limit, cfg.num_rounds, cfg.has_all_in, cfg.bet_sizes = [100], 2, False, [-2.0]  # one increment for two rounds
    with pytest.raises((RuntimeError, ValueError), match="round"):
        cpp.Engine(cfg)


def _net(game, **config):
    import torch

    from headsup.model import BaseModel

    torch.manual_seed(0)
    return native.make_model(BaseModel(game=game, dim=8, **config).numpy_weights())


def test_samplers_and_envs_reject_networks_that_do_not_fit_the_game():
    """A network with fewer outputs than the game has actions left the tail of the output buffer uninitialised
    (used as advantages / baseline values / logits); only run_traversals checked the head size."""
    from headsup.game import DEFAULT_GAME, GameConfig

    cpp = native.module()
    pot = GameConfig(bet_sizes=(0.5, 1.0, 2.0), mask_redundant=True)  # 6 actions
    n4, n6 = _net(DEFAULT_GAME, features="history"), _net(pot, features="history")
    b4, b6 = _net(DEFAULT_GAME, features="history", opp_cards=True), _net(pot, features="history", opp_cards=True)
    cfg6 = native.engine_config(game=pot)
    wrong = (RuntimeError, ValueError)
    with pytest.raises(wrong, match="outputs"):
        cpp.run_dream(n6, n6, b4, 0, 5, 1.0, 0.5, 0, cfg6)
    with pytest.raises(wrong, match="outputs"):
        cpp.run_dream(n4, n4, b6, 0, 5, 1.0, 0.5, 0, cfg6)
    with pytest.raises(wrong, match="outputs"):
        cpp.run_escher_values(n4, n4, 5, 0, cfg6, 0.01)
    with pytest.raises(wrong, match="outputs"):
        cpp.run_escher_regrets(n6, n6, b4, 0, 5, 1.0, 0, cfg6)
    with pytest.raises(wrong, match="opp_cards"):
        cpp.run_traversals(b6, b6, 0, 5, 1.0, 0, cfg6)  # a history-input (86-wide) net as advantage net overflowed the stack
    with pytest.raises(wrong, match="opp_cards"):
        cpp.run_escher_regrets(n6, n6, n6, 0, 5, 1.0, 0, cfg6)  # the value net must read both players' cards
    env = cpp.VecEnv(4, 0, cfg6, True)
    with pytest.raises(wrong, match="outputs"):
        env.set_opponent_model(n4, False)
    env.set_opponent_model(n6, False)
    assert len(cpp.run_dream(n6, n6, b6, 0, 5, 1.0, 0.5, 0, cfg6)) == 8  # the fitting combination still runs


def test_subgame_solver_ignores_range_weight_on_board_blocked_hands():
    """SubgameSolver kept range weight on hands that contain a board card (VectorSolver zeroes them): run() then
    dealt such hands and evaluated 7 cards with a duplicate."""
    cpp = native.module()
    deck = [0, 14, 28, 42, 5, 19, 33, 47, 9]
    e = cpp.Engine()
    e.reset(deck)
    for _ in range(6):
        e.step(1)  # check / call to the river
    a = np.array([x for x in range(52) for y in range(x + 1, 52)])
    b = np.array([y for x in range(52) for y in range(x + 1, 52)])
    blocked = np.isin(a, deck[4:9]) | np.isin(b, deck[4:9])
    sv = cpp.SubgameSolver()
    sv.build(e)
    ones = np.ones(1326, np.float32)
    sv.set_ranges(ones, ones)
    sv.run(20000, 1)
    s = sv.root_strategy()
    assert not (np.abs(s[blocked] - s[blocked][0]).sum(1) > 0).any()  # blocked hands were never dealt, never updated


def test_bindings_reject_out_of_range_arguments():
    """Arguments that index C arrays are checked at the binding (each of these read or wrote out of bounds)."""
    from headsup.game import DEFAULT_GAME

    cpp = native.module()
    bad = (RuntimeError, ValueError)
    e = cpp.Engine()
    with pytest.raises(bad, match="deck"):
        e.reset([0, 1, 2, 3, 4, 5, 6, 7, 7])  # a duplicated card
    with pytest.raises(bad, match="deck"):
        e.reset([0, 1, 2, 3, 4, 5, 6, 7, 52])
    e.reset(list(range(9)))
    with pytest.raises(bad, match="seat"):
        e.observation(5)
    with pytest.raises(bad, match="raise"):
        e.raise_amount(0)
    net = _net(DEFAULT_GAME, features="history")
    with pytest.raises(bad, match="features"):
        net.forward(np.zeros(31, np.float32))  # a history net reads 79 features
    for _ in range(6):
        e.step(1)  # to the river
    sv = cpp.VectorSolver()
    sv.build(e, 50)
    with pytest.raises(bad, match="freeze"):
        sv.freeze(0, 1326, 1)
    with pytest.raises(bad, match="freeze"):
        sv.freeze(0, 0, 9)
    unbuilt = cpp.SubgameSolver()
    unbuilt.set_ranges(np.ones(1326, np.float32), np.ones(1326, np.float32))
    with pytest.raises(bad, match="build"):
        unbuilt.run(10, 0)


def test_network_forward_clamps_index_features_like_the_torch_model():
    """A corrupted observation (card id 400, stage 9) indexed outside the embedding tables in C++; the torch model
    clamps - both must give the same output."""
    import torch

    from headsup.game import DEFAULT_GAME
    from headsup.model import BaseModel

    torch.manual_seed(0)
    model = BaseModel(game=DEFAULT_GAME, dim=8, features="history")
    with torch.no_grad():
        torch.nn.init.normal_(model.action_head.weight, std=0.5)
    net = native.make_model(model.numpy_weights())
    e = native.module().Engine()
    e.reset(list(range(9)))
    obs = np.asarray(e.observation(0)).copy()
    obs[0:3] = (40.0, 9.0, 400.0)  # rank / suit / card of the first hole card
    obs[21] = 9.0
    with torch.no_grad():
        want = model(torch.from_numpy(obs[None])).numpy()[0]
    np.testing.assert_allclose(np.asarray(net.forward(obs)), want, atol=1e-5)


def test_blueprint_abstraction_arguments_are_checked():
    """Edges / centroids of the wrong size overflowed the per-round tables later (in traverse / nearest_centroid);
    card ids outside the deck indexed the evaluator's tables; an unfitted abstraction was dereferenced."""
    from headsup.game import DEFAULT_GAME

    cpp = native.module()
    bad = (RuntimeError, ValueError)
    bp = cpp.TabularBlueprint()
    bp.build(native.engine_config(game=DEFAULT_GAME), 8, 20)
    with pytest.raises(bad, match="fit"):
        bp.bucket(1, 0, 1, [10, 11, 12])  # nothing fitted yet
    with pytest.raises(bad, match="situations"):
        bp.fit_abstraction(0)
    with pytest.raises(bad, match="edges"):
        bp.edges = [[], [0.1] * 20, [0.5] * 7, [0.5] * 7]  # 20 edges for 8 buckets
    with pytest.raises(bad, match="edges"):
        bp.edges = [[], [0.9, 0.1, 0.5, 0.6, 0.7, 0.8, 0.95], [0.5] * 7, [0.5] * 7]  # not sorted
    bp.edges = [[], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] + [[0.5] * 7] * 2
    assert 0 <= bp.bucket(1, 0, 1, [10, 11, 12]) < 8
    for cards in ((0, 60, [10, 11, 12]), (0, 0, [10, 11, 12]), (0, 1, [1, 11, 12]), (0, 1, [10, 11, -3])):
        with pytest.raises(bad, match="cards"):
            bp.bucket(1, *cards)
        with pytest.raises(bad, match="cards"):
            bp.ehs(*cards)
        with pytest.raises(bad, match="cards"):
            cpp.equity_vs_all(cards[0], cards[1], cards[2], 100, 1000, 0, 5)
    table = cpp.TabularBlueprint()
    table.build(native.engine_config(game=DEFAULT_GAME), 8, 20, "table", 10)
    with pytest.raises(bad, match="centroids"):
        table.centroids = [[], [0.5] * 6, [0.5] * 16, [0.5] * 16]  # 3 centroids for 8 buckets
