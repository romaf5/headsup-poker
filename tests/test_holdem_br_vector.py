"""Vectorised hold'em best response (headsup.algos.holdem_br.VectorBestResponse) against the per-board reference."""

import numpy as np
import pytest
import torch

from headsup import native
from headsup.algos.holdem_br import HoldemBestResponse, MixturePolicy, VectorBestResponse, mixture_policy
from headsup.engine import HeadsUpPoker
from headsup.game import FHP
from headsup.model import BaseModel
from headsup.players import RegretMatchingPlayer, TorchPolicyPlayer, regret_matching_torch

pytestmark = pytest.mark.skipif(not native.available(), reason="C++ extension not built")
FLOPS = [(0, 17, 40), (5, 6, 30), (12, 25, 51)]


def _net(seed, game=FHP, std=0.5):
    torch.manual_seed(seed)
    m = BaseModel(game=game, features="history")
    with torch.no_grad():
        torch.nn.init.normal_(m.action_head.weight, std=std)
        torch.nn.init.normal_(m.action_head.bias, std=std)
    return m.eval()


def _fixed_flops(vb, flops):
    def next_cards(boards, k):
        kids = torch.as_tensor(np.array(flops), device=vb.dev)
        return kids, torch.zeros(len(kids), dtype=torch.long, device=vb.dev)

    vb._next_cards = next_cards
    return vb


def test_matches_the_per_board_reference():
    player = TorchPolicyPlayer(_net(0), device="cpu")
    vb = _fixed_flops(VectorBestResponse(mixture_policy(player, "cpu"), FHP, cards=len(FLOPS), chunk=2), FLOPS)
    res = vb.run()
    e = HeadsUpPoker(rng=np.random.default_rng(0), game=FHP)
    e.reset()
    boards = [f + tuple(c for c in range(52) if c not in f)[:2] for f in FLOPS]
    ref = HoldemBestResponse(player, FHP, boards=len(boards)).evaluate_from(e, [np.ones(1326), np.ones(1326)], boards)
    assert res["br_values"] == pytest.approx(ref["br_values"], rel=1e-5)
    assert res["values"] == pytest.approx(ref["values"], rel=1e-5, abs=1e-6)
    assert res["values"][0] == pytest.approx(-res["values"][1], abs=1e-4)


def test_patched_observations_equal_the_engine_observations():
    player = TorchPolicyPlayer(_net(0), device="cpu")
    vb = VectorBestResponse(mixture_policy(player, "cpu"), FHP)
    rng = np.random.default_rng(3)
    for _ in range(20):
        e = HeadsUpPoker(rng=rng, game=FHP)
        e.reset()
        while not e.done and e.stage == 0:
            e.step(int(rng.choice(np.flatnonzero(e.legal_mask()))))
        if e.done:
            continue
        board = torch.as_tensor([list(e.board[:3])])
        rows = vb._obs(e, e.current, board).numpy()
        hand = tuple(sorted(e.hands[e.current]))
        idx = [i for i, (a, b) in enumerate(__import__("headsup.lbr", fromlist=["COMBOS"]).COMBOS) if (a, b) == hand][0]
        np.testing.assert_array_equal(rows[idx], e.observation(e.current))


def test_mixture_values_are_bilinear_in_the_components():
    """The mixture's realisation weights make the profile value the weighted mean of the component
    pairs' values (the SD-CFR average's defining property) - checks the per-component reach math."""
    nets = [[_net(1), _net(2)], [_net(3), _net(4)]]  # [seat][component]
    w = torch.tensor([1.0, 3.0])

    def rm(net, x, legal):
        return regret_matching_torch(net(x), legal.expand(len(x), -1), "uniform")

    def value(policy):
        return _fixed_flops(VectorBestResponse(policy, FHP, cards=2, chunk=2), FLOPS[:2]).run()["values"][0]

    mix = MixturePolicy(lambda s, x, legal: torch.stack([rm(n, x, legal) for n in nets[s]]), w, FHP, "cpu")
    pairs = sum(w[i] * w[j] * value(MixturePolicy(lambda s, x, legal, i=i, j=j: rm(nets[s][(i, j)[s]], x, legal)[None], [1.0], FHP, "cpu"))
                for i in range(2) for j in range(2)) / w.sum() ** 2
    assert value(mix) == pytest.approx(float(pairs), rel=1e-4)
    # a one-hot mixture is that component
    only = MixturePolicy(lambda s, x, legal: torch.stack([rm(n, x, legal) for n in nets[s]]), [0.0, 1.0], FHP, "cpu")
    single = RegretMatchingPlayer([nets[0][1], nets[1][1]], device="cpu")
    assert value(only) == pytest.approx(value(mixture_policy(single, "cpu")), rel=1e-5)


def test_best_response_to_always_call_matches_the_closed_form():
    """Against always-call the best response per hand is: fold / check down / raise pre-flop, then
    bet the flop exactly when ahead - a closed form in the flop's showdown sums.  Checks the chance
    normalisation (values after the flop divided by P(flop misses a hand pair) = 0.7826)."""
    from headsup.lbr import valid_combos
    from headsup.players import make_player

    flops = FLOPS + [(3, 20, 45), (8, 9, 10)]
    s1, s2 = np.zeros(1326), np.zeros(1326)
    for f in flops:
        ok = valid_combos(f)
        d = np.asarray(native.module().BoardTable(list(f), 3).showdown_values(ok.astype(float))) * ok
        s1, s2 = s1 + d, s2 + np.maximum(d, 0)
    norm = len(flops) * (48 * 47 * 46) / (52 * 51 * 50)
    call, raise_ = (100 * s1 + 100 * s2) / norm, (200 * s1 + 100 * s2) / norm
    br1 = np.maximum(call, raise_).sum() / (1326 * 1225)
    br0 = np.maximum(np.maximum(call, raise_), -50.0 * 1225).sum() / (1326 * 1225)
    policy = mixture_policy(make_player("call", game=FHP), "cpu")
    res = _fixed_flops(VectorBestResponse(policy, FHP, cards=len(flops), chunk=3), flops).run()
    assert res["br_values"] == pytest.approx([br0, br1], rel=1e-5)


def test_fhp_cfr_flop_classes_and_hand_permutations():
    """Suit isomorphism of the exact FHP solver: 1755 flop classes whose orbits cover all 22 100 flops once, and
    24 hand permutations that are bijections commuting with card incidence."""
    import numpy as np

    from headsup.algos.fhp_cfr import canonical_flops, hand_permutations
    from headsup.lbr import COMBOS

    reps, w = canonical_flops()
    assert len(reps) == 1755 and w.sum() == 22100 and set(np.unique(w)) <= {4, 6, 12, 24}
    perms = hand_permutations()
    assert perms.shape == (24, 1326) and all(len(set(p)) == 1326 for p in perms)
    assert (perms[0] == np.arange(1326)).all()  # identity first
    combos = np.asarray(COMBOS)
    for p in perms[[5, 17]]:  # sigma(h) keeps ranks, maps suits consistently
        np.testing.assert_array_equal(np.sort(combos[p] % 13, 1), np.sort(combos % 13, 1))


def test_br_streets_follows_the_policy_off_its_own_path():
    """--br-streets 0 against always-call on 3 fixed flops has a closed form: the responder chooses fold / call /
    raise pre-flop and then checks down like the policy; a raise (probability 0 under the policy) must keep the
    check-down value (it was 0 when the off-path continuation used the policy's reach ratio)."""
    import numpy as np
    import torch

    from headsup.algos.holdem_br import _HAND_PAIRS, VectorBestResponse, chance_factor, mixture_policy
    from headsup.game import FHP
    from headsup.players import make_player

    flops = torch.tensor([[0, 14, 27], [5, 18, 44], [12, 25, 38]])
    pol = mixture_policy(make_player("call", game=FHP), torch.device("cpu"), FHP)
    out = {}
    for stages in ([0], None):
        vbr = VectorBestResponse(pol, FHP, cards=3, chunk=8, br_stages=stages)
        vbr._next_cards = lambda boards, k: (flops.repeat(len(boards), 1), torch.arange(len(boards)).repeat_interleave(len(flops)))
        out[stages is None] = vbr.run()["br_values"]
        signs = vbr._sign(flops)
    S = (signs.sum(2).sum(0) / (len(flops) * chance_factor(0, 3))).numpy()  # E[showdown sign] x opponent mass
    sb = np.maximum.reduce([np.full_like(S, -50 * 1225.0), 100 * S, 200 * S]).sum() / _HAND_PAIRS
    bb = np.maximum(100 * S, 200 * S).sum() / _HAND_PAIRS
    np.testing.assert_allclose(out[False], [sb, bb], rtol=1e-5)
    assert out[True][0] >= out[False][0] - 1e-6 and out[True][1] >= out[False][1] - 1e-6  # the full BR is at least as good


def test_fhp_cfr_rejects_other_games():
    import pytest

    from headsup.algos.fhp_cfr import FHPCFR
    from headsup.games.holdem import make_holdem

    with pytest.raises(ValueError, match="FHP"):
        FHPCFR("cpu", game=make_holdem("hulh"))
