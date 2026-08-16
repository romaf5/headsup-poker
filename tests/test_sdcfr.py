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


def _bank(T, seed=0, device="cpu", **config):
    torch.manual_seed(seed)
    nets = [[BaseModel(**config) for _ in range(T)] for _ in range(2)]
    probe = torch.from_numpy(_observations(200, seed=99))
    for seat in nets:  # non-trivial heads so strategies differ; centred so that some rows have no positive advantage
        for m in seat:
            torch.nn.init.normal_(m.action_head.weight, std=0.5)
            with torch.no_grad():
                m.action_head.bias.copy_(-m(probe).mean(0))
    cfg = nets[0][0].config
    return nets, IterateBank.from_state_dicts([[m.state_dict() for m in seat] for seat in nets], device, cfg)


@pytest.mark.parametrize("config", [dict(), dict(rm_fallback="argmax"), dict(features="history", arch="paper", rm_fallback="argmax")])
def test_iterates_match_regret_matching_players(config):
    nets, bank = _bank(4, **config)
    obs = _observations()
    sig = bank.strategies(0, obs)  # every row evaluated with seat-0 nets
    for t in range(4):
        rm = RegretMatchingPlayer([nets[0][t], nets[0][t]], device="cpu")
        np.testing.assert_allclose(sig[t].numpy(), rm.probs(obs), atol=3e-4)  # vmap vs plain kernels; tiny totals amplify fp noise
    assert np.allclose(sig.sum(-1).numpy(), 1.0, atol=1e-5)
    assert np.all(sig[:, obs[:, 23] <= 0, 0].numpy() == 0)  # never fold for free
    if config.get("rm_fallback") == "argmax":  # the fallback rows are one-hot
        with torch.no_grad():
            adv = nets[0][0](torch.from_numpy(obs)).clone()
        adv[torch.from_numpy(obs[:, 23] <= 0), 0] = -1e30
        fb = (adv.clamp(min=0).sum(1) <= 1e-6).numpy()
        assert fb.any()
        assert np.all(sig[0][fb].max(1).values.numpy() == 1.0)


def test_truncated_bank_and_specs(tmp_path):
    from headsup.players import make_player, parse_sdcfr_spec

    assert parse_sdcfr_spec("x/it.pt@exact@g2@t50") == ("x/it.pt", "exact", 2.0, 50, None)
    assert parse_sdcfr_spec("x/it.pt@k32") == ("x/it.pt", "sample", 1.0, None, 32)
    nets, bank = _bank(5)
    bank.save(tmp_path / "it.pt")
    obs = _observations(40)
    short = bank.truncate(3)
    assert short.T == 3 and bank.truncate(9) is bank
    np.testing.assert_allclose(short.strategies(0, obs).numpy(), bank.strategies(0, obs)[:3].numpy())
    p = make_player(f"sdcfr:{tmp_path / 'it.pt'}@t3@exact", device="cpu", seed=0)
    assert p.bank.T == 3 and p.mode == "exact"
    assert make_player(f"sdcfr:{tmp_path / 'it.pt'}@k2", device="cpu", seed=0).bank.T == 2
    it = make_player(f"iterate:{tmp_path / 'it.pt'}@t2", device="cpu", seed=0)
    ref = RegretMatchingPlayer([nets[0][2], nets[1][2]], device="cpu")
    np.testing.assert_allclose(it.probs(obs), ref.probs(obs), atol=1e-6)


def test_bank_save_load(tmp_path):
    _, bank = _bank(3, features="history", rm_fallback="argmax")
    bank.save(tmp_path / "it.pt")
    other = IterateBank.load(tmp_path / "it.pt", "cpu")
    assert other.config == bank.config and other.rm_fallback == "argmax" and other.obs_dim == 79
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
    assert len(r) == 2000 and opp.known.sum() <= 64
