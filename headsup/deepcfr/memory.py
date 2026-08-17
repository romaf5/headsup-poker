"""Reservoir-sampled sample memories living on the training device.

DeepCFR keeps two kinds of memories: per-player *advantage* memories (obs, iteration,
regret vector) and a *strategy* memory (obs, iteration, action distribution).  Both use
reservoir sampling ("Algorithm R") so that, once full, the buffer holds a uniform sample of
everything ever inserted; the original ring buffer silently dropped the oldest samples.

Storage is compact: the first ``OBS_INT_DIM`` observation entries (card features, stage,
position - small integers) are kept as uint8, the rest as float32, so a 31-feature sample
costs 55 + 20 bytes instead of 144 (a 79-feature one 247 + 20).  ``sample`` returns float32
observations exactly equal to the stored ones, on ``sample_device`` (defaults to the storage
device; memories too big for the GPU can live in host RAM with ``device="cpu"`` and be sampled
straight to the training device - see ``prefetch`` for overlapping that with the fit).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from headsup.engine import OBS_DIM
from headsup.enums import NUM_ACTIONS

OBS_INT_DIM = 23  # 7 x (rank+1, suit+1, card+1), stage, first_to_act: all < 256
OBS_INT_MAX = torch.tensor([13, 4, 52] * 7 + [3, 1], dtype=torch.uint8)  # valid upper bounds per column


class ReservoirBuffer:
    def __init__(self, capacity, device, obs_dim=OBS_DIM, target_dim=NUM_ACTIONS, seed=None, sample_device=None, int_dim=OBS_INT_DIM):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.sample_device = torch.device(sample_device) if sample_device is not None else self.device
        self.obs_dim = int(obs_dim)
        self.int_dim = int(int_dim)  # leading observation entries stored as uint8 (0 for games without integer features)
        self.obs_int = torch.zeros((self.capacity, self.int_dim), dtype=torch.uint8, device=self.device)
        self.obs_float = torch.zeros((self.capacity, self.obs_dim - self.int_dim), dtype=torch.float32, device=self.device)
        self.t = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.target = torch.zeros((self.capacity, target_dim), dtype=torch.float32, device=self.device)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def _write(self, rows, obs, t, target):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        self.obs_int[rows] = obs[:, : self.int_dim].to(torch.uint8).to(self.device)
        self.obs_float[rows] = obs[:, self.int_dim :].to(self.device)
        self.t[rows] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
        self.target[rows] = torch.as_tensor(target, dtype=torch.float32).to(self.device)

    def add(self, obs, t, target):
        obs = np.asarray(obs, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32)
        n = len(t)
        if n == 0:
            return
        if obs.shape[1] < self.obs_dim:
            raise ValueError(f"observations have {obs.shape[1]} features, the buffer stores {self.obs_dim}")
        obs = obs[:, : self.obs_dim]
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

    def sample(self, batch_size, generator=None, staging=None):
        if self.sample_device == self.device:
            idx = torch.randint(0, self.size, (batch_size,), device=self.device, generator=generator)
            obs = torch.cat([self.obs_int[idx].to(torch.float32), self.obs_float[idx]], dim=1)
            return obs, self.t[idx], self.target[idx]
        # host-resident storage: gather straight into pinned staging buffers, async copies to the
        # device, and the uint8 -> float32 conversion + concatenation happen there
        idx = torch.from_numpy(self.rng.integers(0, self.size, batch_size))
        parts = staging if staging is not None else self._staging(batch_size)
        for src, dst in zip((self.obs_int, self.obs_float, self.t, self.target), parts):
            torch.index_select(src, 0, idx, out=dst)
        obs_int, obs_float, t, target = (x.to(self.sample_device, non_blocking=True) for x in parts)
        return torch.cat([obs_int.to(torch.float32), obs_float], dim=1), t, target

    def _staging(self, batch_size):
        pin = self.sample_device.type == "cuda"
        return (
            torch.empty((batch_size, self.int_dim), dtype=torch.uint8, pin_memory=pin),
            torch.empty((batch_size, self.obs_dim - self.int_dim), dtype=torch.float32, pin_memory=pin),
            torch.empty((batch_size,), dtype=torch.float32, pin_memory=pin),
            torch.empty((batch_size, self.target.shape[1]), dtype=torch.float32, pin_memory=pin),
        )

    def prefetch(self, batch_size, steps):
        """Iterate over ``steps`` batches; when the storage is not on the sample device the next
        batch is gathered in a background thread (into alternating pinned staging buffers) while
        the current one is being used."""
        if self.sample_device == self.device:
            for _ in range(steps):
                yield self.sample(batch_size)
            return
        n_buf = 3  # a staging buffer is reused only after its device copy has completed (CUDA event)
        staging = [self._staging(batch_size) for _ in range(n_buf)]
        cuda = self.sample_device.type == "cuda"
        events = [torch.cuda.Event() for _ in range(n_buf)] if cuda else [None] * n_buf
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(self.sample, batch_size, None, staging[0])
            for step in range(steps):
                batch = fut.result()
                k = step % n_buf
                if cuda:
                    events[k].record(torch.cuda.current_stream(self.sample_device))
                nxt = (step + 1) % n_buf
                if cuda and step + 1 >= n_buf:
                    events[nxt].synchronize()  # the copy that read staging[nxt] n_buf - 1 steps ago is done
                fut = pool.submit(self.sample, batch_size, None, staging[nxt])
                yield batch

    @property
    def obs(self):
        """float32 view of the stored observations (materialised; for tests / inspection)."""
        return torch.cat([self.obs_int[: self.size].to(torch.float32), self.obs_float[: self.size]], dim=1)

    def state_dict(self):
        return {
            "obs_int": self.obs_int[: self.size].cpu(),
            "obs_float": self.obs_float[: self.size].cpu(),
            "t": self.t[: self.size].cpu(),
            "target": self.target[: self.size].cpu(),
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "size": self.size,
            "seen": self.seen,
        }

    def load_state_dict(self, state):
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"buffer capacity {self.capacity} < saved size {size}")
        if int(state["obs_dim"]) != self.obs_dim:
            raise ValueError(f"saved observations have {state['obs_dim']} features, the buffer stores {self.obs_dim}")
        obs_int = state["obs_int"]
        bad = (obs_int > OBS_INT_MAX).any(dim=1) if self.int_dim == OBS_INT_DIM else torch.zeros(size, dtype=torch.bool)
        if bad.any():  # corrupted rows (seen once: a flipped bit on a non-ECC GPU) - clamp so they cannot crash a fit
            print(f"warning: {int(bad.sum())} of {size:,} saved observations have out-of-range card/stage features; clamped")
            obs_int = torch.minimum(obs_int, OBS_INT_MAX)
        self.obs_int[:size] = obs_int.to(self.device)
        self.obs_float[:size] = state["obs_float"].to(self.device)
        self.t[:size] = state["t"].to(self.device)
        self.target[:size] = state["target"].to(self.device)
        self.size = size
        self.seen = int(state["seen"])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return self


class CircularBuffer:
    """Fixed-capacity FIFO of (history row, action, scalar target) for the DREAM baselines and the
    ESCHER value net (their papers keep the most recent 200 000 samples, not a reservoir)."""

    def __init__(self, capacity, device, obs_dim, seed=None):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.zeros((self.capacity, int(obs_dim)), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((self.capacity,), dtype=torch.long, device=self.device)
        self.target = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.size = 0
        self.head = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, obs, action, target):
        obs = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        action = torch.as_tensor(np.asarray(action).reshape(-1), dtype=torch.long)
        target = torch.as_tensor(np.asarray(target, dtype=np.float32).reshape(-1))
        n = len(action)
        if n == 0:
            return
        if n >= self.capacity:  # keep the newest rows
            obs, action, target = obs[-self.capacity:], action[-self.capacity:], target[-self.capacity:]
            n = self.capacity
        rows = (self.head + torch.arange(n)) % self.capacity
        rows_dev = rows.to(self.device)
        self.obs[rows_dev] = obs.to(self.device)
        self.action[rows_dev] = action.to(self.device)
        self.target[rows_dev] = target.to(self.device)
        self.head = int((self.head + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, n):
        idx = torch.as_tensor(self.rng.integers(0, self.size, size=n), device=self.device)
        return self.obs[idx], self.action[idx], self.target[idx]

    def state_dict(self):
        return {"obs": self.obs[: self.size].cpu(), "action": self.action[: self.size].cpu(), "target": self.target[: self.size].cpu(),
                "head": self.head, "capacity": self.capacity}

    def load_state_dict(self, state):
        n = len(state["action"])
        self.obs[:n] = state["obs"].to(self.device)
        self.action[:n] = state["action"].to(self.device)
        self.target[:n] = state["target"].to(self.device)
        self.size, self.head = n, int(state["head"]) % self.capacity
