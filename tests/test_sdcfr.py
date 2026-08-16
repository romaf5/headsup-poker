import numpy as np
import pytest
import torch

from headsup.engine import HeadsUpPoker
from headsup.env import PokerVecEnv, play_hands
from headsup.model import BaseModel
from headsup.players import AlwaysCallPlayer, RegretMatchingPlayer
from headsup.sdcfr import IterateBank, SDCFRPlayer


def _observations(n=300, seed=0):
    rng = np.random.default_rng(seed)
    e = HeadsUpPoker(rng=rng)
    rows = []
    while len(rows) < n:
        e.reset()
        while not e.done:
            rows.append(e.observation())
            e.step(rng.integers(4))
    return np.stack(rows[:n])


def _bank(T, seed=0, device="cpu"):
    torch.manual_seed(seed)
    nets = [[BaseModel() for _ in range(T)] for _ in range(2)]
    for seat in nets:  # non-trivial heads so strategies differ
        for m in seat:
            torch.nn.init.normal_(m.action_head.weight, std=0.5)
            torch.nn.init.normal_(m.action_head.bias, std=0.5)
    return nets, IterateBank.from_state_dicts([[m.state_dict() for m in seat] for seat in nets], device)


def test_iterates_match_regret_matching_players():
    nets, bank = _bank(4)
    obs = _observations()
    sig = bank.strategies(0, obs)  # every row evaluated with seat-0 nets
    for t in range(4):
        rm = RegretMatchingPlayer([nets[0][t], nets[0][t]], device="cpu")
        np.testing.assert_allclose(sig[t].numpy(), rm.probs(obs), atol=1e-5)
    assert np.allclose(sig.sum(-1).numpy(), 1.0, atol=1e-5)


def test_bank_save_load(tmp_path):
    _, bank = _bank(3)
    bank.save(tmp_path / "it.pt")
    other = IterateBank.load(tmp_path / "it.pt", "cpu")
    obs = _observations(50)
    np.testing.assert_allclose(bank.strategies(1, obs).numpy(), other.strategies(1, obs).numpy())


def test_exact_average_telescopes_to_trajectory_probability():
    """P(own action sequence) under exact reach-weighted averaging == sum_t w_t prod_k sigma_t(I_k, a_k) / sum_t w_t."""
    _, bank = _bank(5, seed=1)
    player = SDCFRPlayer(bank, mode="exact", seed=0)
    e = HeadsUpPoker(rng=np.random.default_rng(3))
    e.reset()
    seat = 0  # we act as the dealer; the opponent always calls
    logp_exact = 0.0
    sig_path = []
    rng = np.random.default_rng(5)
    while not e.done:
        if e.current == seat:
            obs = e.observation()[None]
            p = player.probs(obs, [0])[0]
            choices = np.array([1, 2])  # check/call or raise: keeps the hand going to showdown
            a = int(rng.choice(choices, p=p[choices] / p[choices].sum()))
            logp_exact += np.log(p[a])
            sig_path.append(bank.strategies(seat, obs)[:, 0, a].numpy())
            player.observe(obs, [0], [a])
            e.step(a)
        else:
            e.step(1)
    w = np.arange(1, bank.T + 1, dtype=np.float64)
    prod = np.prod(np.stack(sig_path), axis=0)  # (T,)
    logp_sample = np.log((w * prod).sum() / w.sum())
    assert len(sig_path) >= 2
    assert abs(logp_exact - logp_sample) < 1e-4


def test_sample_and_exact_modes_agree_statistically():
    _, bank = _bank(6, seed=2)
    means = {}
    for mode in ("sample", "exact"):
        player = SDCFRPlayer(bank, mode=mode, seed=0)
        env = PokerVecEnv(256, AlwaysCallPlayer(), seed=0)
        r = play_hands(env, player, 20000)
        means[mode] = (r.mean(), r.std() / np.sqrt(len(r)))
    diff = abs(means["sample"][0] - means["exact"][0])
    assert diff < 4 * (means["sample"][1] + means["exact"][1])


def test_sdcfr_as_opponent_with_subset_ids():
    """The opponent path calls the player with a subset of tables each step."""
    _, bank = _bank(3, seed=4)
    opp = SDCFRPlayer(bank, mode="exact", seed=1)
    env = PokerVecEnv(64, opp, seed=0)
    r = play_hands(env, AlwaysCallPlayer(), 2000)
    assert len(r) == 2000 and len(opp.reach) <= 64
