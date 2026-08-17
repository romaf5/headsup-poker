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

from headsup.engine import legal_mask_from_obs
from headsup.model import BaseModel, normalize_config
from headsup.players import regret_matching_torch, sample_actions


class IterateBank:
    """Stacked parameters of the advantage nets of all iterations, for both seats.

    ``config`` is the nets' variant (see :class:`headsup.model.BaseModel`), including the
    regret-matching fallback the run was trained with.
    """

    def __init__(self, stacked, device, config, weight_power=1.0):
        # stacked: {seat: {param_name: tensor (T, ...)}}
        self.device = torch.device(device)
        self.params = {s: {k: v.to(self.device) for k, v in d.items()} for s, d in stacked.items()}
        self.T = next(iter(self.params[0].values())).shape[0]
        self.set_weight_power(weight_power)
        self.config = normalize_config(config)
        self._base = BaseModel(config=self.config).to(self.device).eval()
        self.dim = self._base.dim
        self.obs_dim = self._base.obs_dim
        self.rm_fallback = self._base.rm_fallback
        self.game = self._base.game
        self.num_actions = self._base.num_actions
        self._fn = lambda params, obs: functional_call(self._base, params, (obs,))
        self._vmapped = vmap(self._fn, in_dims=(0, None))

    def set_weight_power(self, gamma):
        """Iterate t gets weight t^gamma: 1 = linear CFR (default), 2 = DCFR's average-strategy discount."""
        self.weight_power = float(gamma)
        self.weights = torch.arange(1, self.T + 1, dtype=torch.float32, device=self.device) ** self.weight_power

    def truncate(self, n):
        """The bank of the first ``n`` iterates (= the average strategy after n iterations)."""
        if n >= self.T:
            return self
        stacked = {s: {k: v[:n] for k, v in d.items()} for s, d in self.params.items()}
        return IterateBank(stacked, self.device, self.config, self.weight_power)

    def thin(self, k):
        """A bank of ``k`` representative iterates approximating this one's average strategy.

        The iterates are split into ``k`` bins of (nearly) equal total weight; each bin is
        represented by its last iterate, carrying the bin's total weight.  Used where querying
        all T iterates is too expensive (LBR's opponent model); the approximation only weakens
        the best response, so bounds computed against it stay valid.
        """
        if k >= self.T:
            return self
        w = self.weights.double()
        cum = torch.cumsum(w, 0) / w.sum()
        edges = torch.arange(1, k + 1, dtype=torch.float64, device=self.device) / k
        last = torch.searchsorted(cum, edges, right=False).clamp(max=self.T - 1)
        last = torch.unique(last)  # bins never overlap
        first = torch.cat([last.new_zeros(1), last[:-1] + 1])
        bin_w = torch.stack([w[a : b + 1].sum() for a, b in zip(first.tolist(), last.tolist())]).float()
        stacked = {s: {n: v.index_select(0, last) for n, v in d.items()} for s, d in self.params.items()}
        bank = IterateBank(stacked, self.device, self.config, self.weight_power)
        bank.weights = bin_w
        bank.thinned_from = self.T
        return bank

    @staticmethod
    def from_state_dicts(seat_dicts, device, config):
        """seat_dicts: [[state_dict_t0, state_dict_t1, ...] for seat 0, [...] for seat 1]."""
        config = normalize_config(config)
        stacked = {}
        for seat, dicts in enumerate(seat_dicts):
            models = []
            for sd in dicts:
                m = BaseModel(config=config)
                m.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
                models.append(m)
            params, _ = stack_module_state(models)
            stacked[seat] = {k: v.detach() for k, v in params.items()}
        return IterateBank(stacked, device, config)

    def save(self, path):
        torch.save(
            {"seats": {s: {k: v.cpu() for k, v in d.items()} for s, d in self.params.items()}, "T": self.T, "config": dict(self.config)},
            path,
        )

    @staticmethod
    def load(path, device, weight_power=1.0):
        data = torch.load(path, map_location="cpu", weights_only=True)
        return IterateBank({int(s): d for s, d in data["seats"].items()}, device, data["config"], weight_power)

    # rows per vmapped forward: keeps the (T, rows, dim) activations at ~1 GB so big LBR queries
    # (100k+ rows x many iterates) do not run the GPU allocator against its limit
    ACTIVATION_BUDGET = 1 << 30

    @torch.no_grad()
    def strategies(self, seat, obs):
        """Regret-matched strategies of every iterate: (T, B, A); illegal actions masked."""
        obs = np.asarray(obs, dtype=np.float32)
        x = torch.as_tensor(obs[:, : self.obs_dim]).to(self.device)
        chunk = max(256, self.ACTIVATION_BUDGET // (self.T * self.dim * 4 * 12))
        outs = []
        for i in range(0, len(x), chunk):
            xi = x[i : i + chunk]
            try:
                adv = self._vmapped(self.params[seat], xi)
            except Exception:  # vmap unsupported op on this backend: fall back to a loop
                adv = torch.stack([self._fn({k: v[t] for k, v in self.params[seat].items()}, xi) for t in range(self.T)])
            outs.append(regret_matching_torch(adv, legal_mask_from_obs(obs[i : i + chunk], self.game), self.rm_fallback))
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


class SDCFRPlayer:
    """Plays the SD-CFR average strategy.  Batched; tracks per-table state via ``ids``.

    State lives in dense tensors indexed by table id (``reach``: (capacity, T) own reach under
    each iterate for exact mode; ``chosen``: the iterate sampled for the current hand in sample
    mode), grown on demand - ids can be any non-negative ints (LBR uses ``table * 1326 + combo``).
    """

    wants_ids = True

    def __init__(self, bank, mode="sample", seed=None, weight_power=None):
        assert mode in ("sample", "exact")
        self.bank = bank
        if weight_power is not None:
            bank.set_weight_power(weight_power)
        self.mode = mode
        self.game = bank.game
        self.rng = np.random.default_rng(seed)
        dev = self.bank.device
        self.reach = torch.empty((0, self.bank.T), device=dev)  # own reach under each iterate (exact mode)
        self.chosen = torch.empty((0,), dtype=torch.long, device=dev)  # sampled iterate per table (sample mode)
        self.known = np.zeros(0, dtype=bool)  # ids that have started a hand
        self.last_probs = None
        w = self.bank.weights.cpu().numpy().astype(np.float64)
        self._w = w / w.sum()

    def _grow(self, max_id):
        if max_id < len(self.known):
            return
        cap = max(2 * len(self.known), int(max_id) + 1, 1024)
        reach = torch.ones((cap, self.bank.T), device=self.bank.device)
        reach[: len(self.known)] = self.reach
        chosen = torch.zeros((cap,), dtype=torch.long, device=self.bank.device)
        chosen[: len(self.known)] = self.chosen
        known = np.zeros(cap, dtype=bool)
        known[: len(self.known)] = self.known
        self.reach, self.chosen, self.known = reach, chosen, known

    @staticmethod
    def load(path, device, mode="sample", seed=None, weight_power=1.0, iterations=None, thin=None):
        bank = IterateBank.load(path, device, weight_power)
        if iterations:
            bank = bank.truncate(iterations)
        if thin:
            bank = bank.thin(thin)
        return SDCFRPlayer(bank, mode=mode, seed=seed)

    # ------------------------------------------------------------------ state
    @staticmethod
    def _first_decision(obs):
        """True for rows at the player's first decision of a hand: pre-flop with no action taken yet
        (the small blind, obs[22] == 0) or exactly the small blind's opening action (the big blind).
        Read from the bet history, so it does not depend on the blind sizes."""
        from headsup.engine import HISTORY_SLOTS, history_slot

        stage = obs[:, 21]
        occurred = obs[:, history_slot(0, 0) + 1 : history_slot(0, HISTORY_SLOTS - 1) + 2 : 2]
        n_preflop = occurred.sum(axis=1)
        return (stage == 0) & (n_preflop == obs[:, 22])

    def start_hand(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return
        self._grow(ids.max())
        idx = torch.as_tensor(ids, device=self.bank.device)
        self.reach[idx] = 1.0
        self.chosen[idx] = torch.as_tensor(self.rng.choice(self.bank.T, size=len(ids), p=self._w), device=self.bank.device)
        self.known[ids] = True

    def _ensure_state(self, obs, ids):
        ids = np.asarray(ids, dtype=np.int64)
        self._grow(ids.max())
        new = ids[self._first_decision(obs) | ~self.known[ids]]
        if len(new):
            self.start_hand(new)

    # ------------------------------------------------------------------ queries
    def _seat_split(self, obs):
        seat = np.asarray(obs)[:, 22].astype(int)  # 0 = dealer / seat 0, 1 = big blind / seat 1
        return seat

    def strategies(self, obs):
        """(T, B, 4) iterate strategies for rows of possibly mixed seats."""
        obs = np.asarray(obs, dtype=np.float32)
        seat = self._seat_split(obs)
        out = torch.empty((self.bank.T, len(obs), self.bank.num_actions), device=self.bank.device)
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
        idx = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=self.bank.device)
        if self.mode == "sample":
            p = sig[self.chosen[idx], torch.arange(len(ids), device=self.bank.device)]
        else:
            w = (self.reach[idx] * self.bank.weights).T.unsqueeze(-1)  # (T, B, 1)
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
        idx = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=self.bank.device)
        self.reach[idx] *= taken.T
        self._last_sig = None

    def __call__(self, obs, ids=None):
        obs = np.asarray(obs, dtype=np.float32)
        ids = np.arange(len(obs)) if ids is None else np.asarray(ids)
        self.last_probs = self.probs(obs, ids)
        actions = sample_actions(self.last_probs, self.rng)
        self.observe(obs, ids, actions)
        return actions
