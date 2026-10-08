"""Refit noise of the regret nets on FHP's 1,352 pre-flop infosets (169 classes x 8 nodes), from a saved iterate bank."""
import sys, numpy as np, torch
from headsup.engine import HeadsUpPoker, legal_mask_from_obs
from headsup.game import FHP
from headsup.sdcfr import IterateBank
from headsup.players import regret_matching_torch

def regret_matching_np(adv, legal, fallback):
    return regret_matching_torch(torch.as_tensor(adv), torch.as_tensor(legal), fallback).numpy().astype(np.float64)

torch.set_num_threads(4)
bank = IterateBank.load(sys.argv[1], "cpu")
last = int(sys.argv[2]) if len(sys.argv) > 2 else None
if last is not None:
    bank.truncate(last)
T = bank.T
# one representative combo per hand class: pairs (r, r), suited (hi, lo) same suit, offsuit
reps, names = [], []
for hi in range(13):
    for lo in range(hi + 1):
        if hi == lo:
            reps.append((hi, lo + 13)); names.append("pair")
        else:
            reps.append((hi, lo)); names.append("suited")
            reps.append((hi, lo + 13)); names.append("offsuit")
combos = {"pair": 6, "suited": 4, "offsuit": 12}
w_hand = np.array([combos[n] for n in names], dtype=np.float64); w_hand /= w_hand.sum()
SEQS = ["", "c", "cr", "crr", "crrr", "r", "rr", "rrr"]
obs = {0: [], 1: []}; meta = {0: [], 1: []}
for seq in SEQS:
    seat = len(seq) % 2
    for k, (a, b) in enumerate(reps):
        rest = [c for c in range(52) if c not in (a, b)]
        hands = [(a, b), (rest[0], rest[1])] if seat == 0 else [(rest[0], rest[1]), (a, b)]
        deck = list(hands[0]) + list(hands[1]) + rest[2:7]
        e = HeadsUpPoker(game=FHP); e.reset(deck)
        for ch in seq:
            e.step(1 if ch == "c" else 2)
        assert e.current == seat and not e.done and e.stage == 0
        obs[seat].append(e.observation()); meta[seat].append((seq, k))
for seat in (0, 1):
    o = np.stack(obs[seat]).astype(np.float32)
    legal = legal_mask_from_obs(o, FHP)
    with torch.no_grad():
        adv = bank._vmapped(bank.params[seat], torch.as_tensor(o[:, : bank.obs_dim])).numpy().astype(np.float64)  # (T, B, A)
    adv = np.where(legal[None], adv, np.nan)
    sig = np.stack([regret_matching_np(np.nan_to_num(adv[t], nan=-1e9).astype(np.float32), legal, bank.rm_fallback) for t in range(T)])
    w = np.tile(w_hand, len(o) // 169)[:, None] * legal; w = w / w.sum()
    win = range(T - 11, T)  # the last 11 iterates
    A = adv[list(win)]
    mean_adv = np.nanmean(A, axis=0)
    rms = lambda x: float(np.sqrt(np.nansum(w * x ** 2)))
    step = np.mean([rms(adv[t] - adv[t - 1]) for t in win[1:]])
    within = float(np.sqrt(np.nansum(w * np.nanvar(A, axis=0))))
    S = sig[list(win)]
    rm_of_mean = regret_matching_np(np.nan_to_num(mean_adv, nan=-1e9).astype(np.float32), legal, bank.rm_fallback)
    wi = np.tile(w_hand, len(o) // 169); wi = wi / wi.sum()
    l1_step = np.mean([(wi * np.abs(sig[t] - sig[t - 1]).sum(1)).sum() for t in win[1:]])
    l1_bias = (wi * np.abs(S.mean(0) - rm_of_mean).sum(1)).sum()
    neg_mass = (wi * (S.mean(0) * (np.nan_to_num(mean_adv, nan=0.0) < 0)).sum(1)).sum()
    pure = (wi * (S.max(-1) > 0.999).mean(0)).sum()
    print(f"seat {seat}: {len(o)} infosets, iterates {bank.iterations[T - 11].item()}..{bank.iterations[-1].item()}")
    print(f"  predicted advantages (chips; big blind = 100): rms {rms(mean_adv):.1f}; between consecutive refits rms {step:.1f}; std over the 11 nets {within:.1f}")
    print(f"  strategies: L1 between consecutive iterates {l1_step:.3f}; mass on actions whose mean advantage is negative {neg_mass:.3f}; "
          f"L1(mean of the 11 strategies, RM of the mean advantage) {l1_bias:.3f}; pure-strategy share {pure:.2f}")
