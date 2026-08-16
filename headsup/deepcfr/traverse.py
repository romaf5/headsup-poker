"""External-sampling MCCFR traversals for DeepCFR.

``traverse`` is the readable Python reference (numpy network, Python engine).  The C++
extension implements the identical algorithm ~50x faster and releases the GIL, so
:class:`TraversalRunner` runs it from a thread pool; without the extension it falls back to
a process pool running the Python version.

Sample semantics (per DeepCFR, Brown et al. 2019):
* at the traverser's decision nodes every action is explored; the sample stored in the
  advantage memory is ``(obs, t, v(a) - sum_a sigma(a) v(a))``;
* at the opponent's nodes one action is sampled from the opponent's regret-matched
  strategy, which is stored in the strategy memory as ``(obs, t, sigma)``.
Iteration weights ``t`` implement linear CFR.
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from headsup.engine import OBS_DIM, HeadsUpPoker
from headsup.enums import NUM_ACTIONS
from headsup.numpy_model import NumpyModel


def regret_matching(adv, eps=1e-6):
    pos = np.clip(adv, 0.0, None)
    total = pos.sum()
    if total <= eps:
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    return (pos / total).astype(np.float32)


@dataclass
class Samples:
    obs: np.ndarray
    t: np.ndarray
    target: np.ndarray

    def __len__(self):
        return len(self.t)

    @staticmethod
    def empty():
        return Samples(
            np.zeros((0, OBS_DIM), np.float32), np.zeros((0,), np.float32), np.zeros((0, NUM_ACTIONS), np.float32)
        )

    @staticmethod
    def concat(items):
        items = [s for s in items if len(s)]
        if not items:
            return Samples.empty()
        return Samples(
            np.concatenate([s.obs for s in items]),
            np.concatenate([s.t for s in items]),
            np.concatenate([s.target for s in items]),
        )


class _Memory:
    def __init__(self):
        self.obs, self.t, self.target = [], [], []

    def add(self, obs, t, target):
        self.obs.append(obs)
        self.t.append(t)
        self.target.append(target)

    def to_samples(self):
        if not self.t:
            return Samples.empty()
        return Samples(
            np.stack(self.obs).astype(np.float32),
            np.asarray(self.t, dtype=np.float32),
            np.stack(self.target).astype(np.float32),
        )


def traverse(engine, traverser, nets, t, rng, adv_mem, strat_mem, stats=None):
    """Recursive external-sampling traversal; returns the traverser's expected value."""
    if engine.done:
        return float(engine.rewards[traverser])
    p = engine.current
    obs = engine.observation()
    sigma = regret_matching(nets[p](obs))
    if stats is not None:
        stats["nodes"] += 1
    if p == traverser:
        values = np.empty(NUM_ACTIONS, dtype=np.float32)
        for a in range(NUM_ACTIONS):
            child = engine.clone() if a + 1 < NUM_ACTIONS else engine
            child.step(a)
            values[a] = traverse(child, traverser, nets, t, rng, adv_mem, strat_mem, stats)
        mean = float(np.dot(sigma, values))
        adv_mem.add(obs, t, values - mean)
        return mean
    strat_mem.add(obs, t, sigma)
    a = int(np.searchsorted(np.cumsum(sigma), rng.random(), side="right"))
    engine.step(min(a, NUM_ACTIONS - 1))
    return traverse(engine, traverser, nets, t, rng, adv_mem, strat_mem, stats)


def run_traversals_python(weights, traverser, n_traversals, t, seed, engine_kwargs=None, decks=None):
    """Worker entry point (picklable): returns (adv Samples, strat Samples, nodes)."""
    rng = np.random.default_rng(seed)
    engine = HeadsUpPoker(rng=rng, **(engine_kwargs or {}))
    nets = [NumpyModel(weights[0]), NumpyModel(weights[1])]
    adv, strat, stats = _Memory(), _Memory(), {"nodes": 0}
    for i in range(n_traversals):
        engine.reset(None if decks is None else decks[i])
        traverse(engine, traverser, nets, t, rng, adv, strat, stats)
    return adv.to_samples(), strat.to_samples(), stats["nodes"]


class TraversalRunner:
    """Collects samples for one CFR iteration using all CPU cores."""

    def __init__(self, num_workers=None, backend="auto", engine_kwargs=None):
        self.num_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
        self.engine_kwargs = engine_kwargs or {}
        if backend == "auto":
            from headsup import native

            backend = "cpp" if native.available() else "python"
        self.backend = backend
        self._pool = None
        if backend == "cpp":
            from headsup import native

            self._cpp = native.module()
            self._cfg = native.engine_config(**self.engine_kwargs)
            self._pool = ThreadPoolExecutor(self.num_workers)
        elif backend == "python":
            import multiprocessing as mp

            self._pool = ProcessPoolExecutor(self.num_workers, mp_context=mp.get_context("spawn"))
        else:
            raise ValueError(backend)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _split(self, n):
        per = [n // self.num_workers] * self.num_workers
        for i in range(n % self.num_workers):
            per[i] += 1
        return [k for k in per if k > 0]

    def collect(self, weights, traverser, n_traversals, t, seed):
        """``weights``: numpy state dicts for seat 0 and seat 1.  Returns (adv, strat, nodes)."""
        chunks = self._split(n_traversals)
        seeds = np.random.SeedSequence(seed).generate_state(len(chunks), dtype=np.uint64)
        if self.backend == "cpp":
            from headsup import native

            nets = [native.make_model(weights[0]), native.make_model(weights[1])]
            futs = [
                self._pool.submit(self._cpp.run_traversals, nets[0], nets[1], traverser, k, float(t), int(s), self._cfg)
                for k, s in zip(chunks, seeds)
            ]
            outs = [f.result() for f in futs]
            adv = Samples.concat([Samples(o[0], o[1], o[2]) for o in outs])
            strat = Samples.concat([Samples(o[3], o[4], o[5]) for o in outs])
            nodes = sum(o[6] for o in outs)
        else:
            futs = [
                self._pool.submit(
                    run_traversals_python, weights, traverser, k, float(t), int(s), self.engine_kwargs
                )
                for k, s in zip(chunks, seeds)
            ]
            outs = [f.result() for f in futs]
            adv = Samples.concat([o[0] for o in outs])
            strat = Samples.concat([o[1] for o in outs])
            nodes = sum(o[2] for o in outs)
        return adv, strat, nodes
