"""Reservoir-sampled sample memories living on the training device.

DeepCFR keeps two kinds of memories: per-player *advantage* memories (obs, iteration,
regret vector) and a *strategy* memory (obs, iteration, action distribution).  Both use
reservoir sampling ("Algorithm R") so that, once full, the buffer holds a uniform sample of
everything ever inserted; the original ring buffer silently dropped the oldest samples.
"""

import numpy as np
import torch

from headsup.engine import OBS_DIM
from headsup.enums import NUM_ACTIONS


class ReservoirBuffer:
    def __init__(self, capacity, device, obs_dim=OBS_DIM, target_dim=NUM_ACTIONS, seed=None):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.t = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.target = torch.zeros((self.capacity, target_dim), dtype=torch.float32, device=self.device)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def _write(self, rows, obs, t, target):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self.obs[rows] = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        self.t[rows] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
        self.target[rows] = torch.as_tensor(target, dtype=torch.float32).to(self.device)

    def add(self, obs, t, target):
        obs = np.asarray(obs, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32)
        n = len(t)
        if n == 0:
            return
        k = min(self.capacity - self.size, n)  # fills empty slots first
        if k > 0:
            rows = np.arange(self.size, self.size + k)
            self._write(rows, obs[:k], t[:k], target[:k])
            self.size += k
        if n > k:
            # item with 1-based global index m is kept with prob capacity / m at a uniform slot
            m = self.seen + np.arange(k, n) + 1
            r = self.rng.random(n - k) * m
            keep = r < self.capacity
            if keep.any():
                slots = r[keep].astype(np.int64)
                sel = np.flatnonzero(keep) + k
                self._write(slots, obs[sel], t[sel], target[sel])
        self.seen += n

    def sample(self, batch_size, generator=None):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device, generator=generator)
        return self.obs[idx], self.t[idx], self.target[idx]

    def state_dict(self):
        return {
            "obs": self.obs[: self.size].cpu(),
            "t": self.t[: self.size].cpu(),
            "target": self.target[: self.size].cpu(),
            "capacity": self.capacity,
            "size": self.size,
            "seen": self.seen,
        }

    def load_state_dict(self, state):
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"buffer capacity {self.capacity} < saved size {size}")
        self.obs[:size] = state["obs"].to(self.device)
        self.t[:size] = state["t"].to(self.device)
        self.target[:size] = state["target"].to(self.device)
        self.size = size
        self.seen = int(state["seen"])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return self
