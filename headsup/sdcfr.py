"""Single Deep CFR (Steinberger, 2019): the average strategy without a strategy network.

DeepCFR approximates the average strategy with a network fitted on a strategy memory.
SD-CFR instead keeps the advantage network of *every* iteration and derives the linear-CFR
average strategy directly from them at play time:

* **exact** mode: at each own decision, sigma_bar(I) = sum_t w_t * pi_t(I) * sigma_t(I) /
  sum_t w_t * pi_t(I), where sigma_t = regret matching on iterate t, w_t = t and pi_t(I) is
  the player's own reach probability of I under sigma_t (a running product over the own
  decisions of the current hand, so it needs per-table state);
* **sample** mode: draw one iteration t ~ w_t at the start of each hand and follow sigma_t
  for the whole hand.  This "trajectory sampling" induces exactly the same distribution
  over the player's action sequences (the reach-weighted average telescopes), at the cost
  of one gather instead of a weighted sum.

All iterates are evaluated with a single vmapped forward pass ((T, B, 4) advantages).
"""

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from headsup.enums import NUM_ACTIONS
from headsup.model import BaseModel
from headsup.players import sample_actions


def _regret_matching(adv):
    pos = adv.clamp(min=0.0)
    total = pos.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(pos, 1.0 / NUM_ACTIONS)
    return torch.where(total > 1e-6, pos / total.clamp(min=1e-6), uniform)


class IterateBank:
    """Stacked parameters of the advantage nets of all iterations, for both seats."""

    def __init__(self, stacked, device):
        # stacked: {seat: {param_name: tensor (T, ...)}}
        self.device = torch.device(device)
        self.params = {s: {k: v.to(self.device) for k, v in d.items()} for s, d in stacked.items()}
        self.T = next(iter(self.params[0].values())).shape[0]
        self.weights = torch.arange(1, self.T + 1, dtype=torch.float32, device=self.device)  # linear CFR
        self._base = BaseModel().to(self.device).eval()
        self._fn = lambda params, obs: functional_call(self._base, params, (obs,))
        self._vmapped = vmap(self._fn, in_dims=(0, None))

    @staticmethod
    def from_state_dicts(seat_dicts, device):
        """seat_dicts: [[state_dict_t0, state_dict_t1, ...] for seat 0, [...] for seat 1]."""
        stacked = {}
        for seat, dicts in enumerate(seat_dicts):
            models = []
            for sd in dicts:
                m = BaseModel()
                m.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
                models.append(m)
            params, buffers = stack_module_state(models)
            stacked[seat] = {k: v.detach() for k, v in {**params, **buffers}.items()}
        return IterateBank(stacked, device)

    def save(self, path):
        torch.save({"seats": {s: {k: v.cpu() for k, v in d.items()} for s, d in self.params.items()}, "T": self.T}, path)

    @staticmethod
    def load(path, device):
        data = torch.load(path, map_location="cpu", weights_only=True)
        return IterateBank({int(s): d for s, d in data["seats"].items()}, device)

    @torch.no_grad()
    def strategies(self, seat, obs):
        """Regret-matched strategies of every iterate: (T, B, 4)."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
        try:
            adv = self._vmapped(self.params[seat], x)
        except Exception:  # vmap unsupported op on this backend: fall back to a loop
            adv = torch.stack([self._fn({k: v[t] for k, v in self.params[seat].items()}, x) for t in range(self.T)])
        return _regret_matching(adv)


class SDCFRPlayer:
    """Plays the SD-CFR average strategy.  Batched; tracks per-table state via ``ids``."""

    wants_ids = True

    def __init__(self, bank, mode="sample", seed=None, small_blind=1, big_blind=2):
        assert mode in ("sample", "exact")
        self.bank = bank
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.small_blind, self.big_blind = small_blind, big_blind
        self.reach = {}  # table id -> torch (T,) own reach under each iterate (exact mode)
        self.chosen = {}  # table id -> sampled iterate index (sample mode)
        self.last_probs = None
        w = self.bank.weights.cpu().numpy().astype(np.float64)
        self._w = w / w.sum()

    @staticmethod
    def load(path, device, mode="sample", seed=None):
        return SDCFRPlayer(IterateBank.load(path, device), mode=mode, seed=seed)

    # ------------------------------------------------------------------ state
    def _first_decision(self, obs):
        """True for rows at the player's first decision of a hand (pre-flop, street bet == own blind)."""
        stage = obs[:, 21]
        my_bet = np.rint(obs[:, 26] * obs[:, 29] * 1000)
        blind = np.where(obs[:, 22] == 0, self.small_blind, self.big_blind)
        return (stage == 0) & (my_bet == blind)

    def start_hand(self, ids):
        for i in ids:
            i = int(i)
            self.reach[i] = torch.ones(self.bank.T, device=self.bank.device)
            self.chosen[i] = int(self.rng.choice(self.bank.T, p=self._w))

    def _ensure_state(self, obs, ids):
        first = self._first_decision(obs)
        new = [int(i) for i, f in zip(ids, first) if f or int(i) not in self.reach]
        if new:
            self.start_hand(new)

    # ------------------------------------------------------------------ queries
    def _seat_split(self, obs):
        seat = np.asarray(obs)[:, 22].astype(int)  # 0 = dealer / seat 0, 1 = big blind / seat 1
        return seat

    def strategies(self, obs):
        """(T, B, 4) iterate strategies for rows of possibly mixed seats."""
        obs = np.asarray(obs, dtype=np.float32)
        seat = self._seat_split(obs)
        out = torch.empty((self.bank.T, len(obs), NUM_ACTIONS), device=self.bank.device)
        for s in (0, 1):
            m = seat == s
            if m.any():
                out[:, torch.as_tensor(np.flatnonzero(m), device=self.bank.device)] = self.bank.strategies(s, obs[m])
        return out

    def probs(self, obs, ids=None):
        obs = np.asarray(obs, dtype=np.float32)
        ids = np.arange(len(obs)) if ids is None else np.asarray(ids)
        self._ensure_state(obs, ids)
        sig = self.strategies(obs)  # (T, B, 4)
        if self.mode == "sample":
            t = torch.as_tensor([self.chosen[int(i)] for i in ids], device=self.bank.device)
            p = sig[t, torch.arange(len(ids), device=self.bank.device)]
        else:
            reach = torch.stack([self.reach[int(i)] for i in ids])  # (B, T)
            w = (reach * self.bank.weights).T.unsqueeze(-1)  # (T, B, 1)
            p = (w * sig).sum(0) / w.sum(0).clamp(min=1e-12)
        self._last_sig = sig
        return p.float().cpu().numpy()

    def observe(self, obs, ids, actions):
        """Update own reach with the actions actually taken (exact mode; no-op in sample mode)."""
        if self.mode != "exact":
            return
        obs = np.asarray(obs, dtype=np.float32)
        sig = self._last_sig if getattr(self, "_last_sig", None) is not None and self._last_sig.shape[1] == len(obs) else self.strategies(obs)
        a = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=self.bank.device)
        taken = sig[:, torch.arange(len(a), device=self.bank.device), a]  # (T, B)
        for j, i in enumerate(ids):
            self.reach[int(i)] = self.reach[int(i)] * taken[:, j]
        self._last_sig = None

    def __call__(self, obs, ids=None):
        obs = np.asarray(obs, dtype=np.float32)
        ids = np.arange(len(obs)) if ids is None else np.asarray(ids)
        self.last_probs = self.probs(obs, ids)
        actions = sample_actions(self.last_probs, self.rng)
        self.observe(obs, ids, actions)
        return actions
