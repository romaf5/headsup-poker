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
from headsup.game import DEFAULT_GAME
from headsup.numpy_model import NumpyModel


def regret_matching(adv, eps=1e-6, fold_allowed=True, fallback="uniform", legal=None):
    """Regret matching over the legal actions (``legal`` bool mask; default: all but FOLD unless
    ``fold_allowed``); ``fallback`` when no legal advantage is positive: ``uniform`` over the
    legal actions, or ``argmax`` = the highest legal advantage (DeepCFR paper)."""
    adv = np.asarray(adv, dtype=np.float32)
    if legal is None:
        legal = np.ones(len(adv), dtype=bool)
        legal[0] = fold_allowed
    legal = np.asarray(legal, dtype=bool)
    pos = np.where(legal, np.clip(adv, 0.0, None), 0.0)
    total = pos.sum()
    if total <= eps:
        sigma = np.zeros(len(adv), dtype=np.float32)
        if fallback == "argmax":
            sigma[int(np.argmax(np.where(legal, adv, -np.inf)))] = 1.0
        else:
            sigma[legal] = 1.0 / legal.sum()
        return sigma
    return (pos / total).astype(np.float32)


@dataclass
class Samples:
    obs: np.ndarray
    t: np.ndarray
    target: np.ndarray

    def __len__(self):
        return len(self.t)

    @staticmethod
    def empty(obs_dim=OBS_DIM, num_actions=DEFAULT_GAME.num_actions):
        return Samples(
            np.zeros((0, obs_dim), np.float32), np.zeros((0,), np.float32), np.zeros((0, num_actions), np.float32)
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
    def __init__(self, obs_dim=OBS_DIM, num_actions=DEFAULT_GAME.num_actions):
        self.obs_dim, self.num_actions = obs_dim, num_actions
        self.obs, self.t, self.target = [], [], []

    def add(self, obs, t, target):
        self.obs.append(obs[: self.obs_dim])
        self.t.append(t)
        self.target.append(target)

    def to_samples(self):
        if not self.t:
            return Samples.empty(self.obs_dim, self.num_actions)
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
    n = engine.num_actions
    obs = engine.observation()
    legal, twin = engine.legal_mask_and_twins()
    sigma = regret_matching(nets[p](obs), legal=legal, fallback=nets[p].rm_fallback)
    if stats is not None:
        stats["nodes"] += 1
    if p == traverser:
        values = np.empty(n, dtype=np.float32)
        for a in range(n):
            if not legal[a]:
                continue  # duplicates another action; filled in below
            child = engine.clone() if a + 1 < n else engine
            child.step(a)
            values[a] = traverse(child, traverser, nets, t, rng, adv_mem, strat_mem, stats)
        for a in range(n):
            if not legal[a]:
                values[a] = values[twin[a]]
        mean = float(np.dot(sigma, values))
        adv_mem.add(obs, t, values - mean)
        return mean
    strat_mem.add(obs, t, sigma)
    a = int(np.searchsorted(np.cumsum(sigma), rng.random(), side="right"))
    engine.step(min(a, n - 1))
    return traverse(engine, traverser, nets, t, rng, adv_mem, strat_mem, stats)


def history_rows(engine):
    """History input of the value nets: seat 0's observation + seat 1's hole cards (see headsup.model)."""
    from headsup.model import history_observation

    return history_observation(engine.observation(0)[None], np.array([engine.hands[1]]))[0]


def dream_trajectory(engine, traverser, nets, baseline, t, epsilon, rng, own_reach, adv_mem, val_mem, stats=None):
    """Python reference of the C++ DREAM sampler (one outcome-sampled trajectory); returns the
    baseline-corrected value of the state for the traverser (DREAM eq. 6-7)."""
    if engine.done:
        return float(engine.rewards[traverser])
    p = engine.current
    n = engine.num_actions
    obs = engine.observation()
    legal, _ = engine.legal_mask_and_twins()
    legal = np.asarray(legal, dtype=bool)
    sigma = np.asarray(regret_matching(nets[p](obs), legal=legal, fallback=nets[p].rm_fallback), dtype=np.float64)
    if stats is not None:
        stats["nodes"] += 1
    xi = (epsilon * legal / legal.sum() + (1.0 - epsilon) * sigma) if p == traverser else sigma
    a = min(int(np.searchsorted(np.cumsum(xi), rng.random(), side="right")), n - 1)
    hist = history_rows(engine)
    b = baseline(hist)
    child = engine.clone()
    child.step(a)
    v_child = dream_trajectory(child, traverser, nets, baseline, t, epsilon, rng, own_reach * (xi[a] if p == traverser else 1.0),
                               adv_mem, val_mem, stats)
    va = np.where(legal, b, 0.0)
    va[a] = b[a] + (v_child - b[a]) / max(xi[a], 1e-12)
    v = float(np.dot(sigma, va))
    if p == traverser:
        adv_mem.add(obs, t / max(own_reach, 1e-12), np.where(legal, va - v, 0.0))
    if child.done:
        q_target = float(child.rewards[traverser])
    else:
        clegal = np.asarray(child.legal_mask_and_twins()[0], dtype=bool)
        csig = regret_matching(nets[child.current](child.observation()), legal=clegal, fallback=nets[child.current].rm_fallback)
        q_target = float(np.dot(csig, np.where(clegal, baseline(history_rows(child)), 0.0)))
    row = np.zeros(n, np.float32)
    row[a] = q_target
    val_mem.add(hist, float(a), row)
    return v


def run_dream_python(weights, baseline_weights, traverser, n_traversals, t, epsilon, seed, engine_kwargs=None, decks=None):
    rng = np.random.default_rng(seed)
    nets = [NumpyModel(weights[0]), NumpyModel(weights[1])]
    baseline = NumpyModel(baseline_weights)
    engine = HeadsUpPoker(rng=rng, game=nets[0].game.with_(**(engine_kwargs or {})))
    from headsup.model import OBS_DIM_WITH_OPP

    adv, val, stats = _Memory(nets[0].obs_dim, engine.num_actions), _Memory(OBS_DIM_WITH_OPP, engine.num_actions), {"nodes": 0}
    for i in range(n_traversals):
        engine.reset(None if decks is None else decks[i])
        dream_trajectory(engine, traverser, nets, baseline, t, epsilon, rng, 1.0, adv, val, stats)
    return adv.to_samples(), val.to_samples(), stats["nodes"]


def run_traversals_python(weights, traverser, n_traversals, t, seed, engine_kwargs=None, decks=None):
    """Worker entry point (picklable): returns (adv Samples, strat Samples, nodes)."""
    rng = np.random.default_rng(seed)
    nets = [NumpyModel(weights[0]), NumpyModel(weights[1])]
    if nets[0].obs_dim != nets[1].obs_dim:
        raise ValueError("both networks must read the same observation width")
    engine = HeadsUpPoker(rng=rng, game=nets[0].game.with_(**(engine_kwargs or {})))
    if engine.num_actions != nets[0].num_actions or engine.num_actions != nets[1].num_actions:
        raise ValueError("the networks' action heads do not match the game's number of actions")
    adv, strat, stats = _Memory(nets[0].obs_dim, engine.num_actions), _Memory(nets[0].obs_dim, engine.num_actions), {"nodes": 0}
    for i in range(n_traversals):
        engine.reset(None if decks is None else decks[i])
        traverse(engine, traverser, nets, t, rng, adv, strat, stats)
    return adv.to_samples(), strat.to_samples(), stats["nodes"]


class TraversalRunner:
    """Collects samples for one CFR iteration using all CPU cores."""

    def __init__(self, num_workers=None, backend="auto", engine_kwargs=None, game=None):
        self.num_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
        self.game = (game or DEFAULT_GAME).with_(**(engine_kwargs or {}))
        self.engine_kwargs = self.game.to_dict()
        if backend == "auto":
            from headsup import native

            backend = "cpp" if native.available() else "python"
        self.backend = backend
        self._pool = None
        if backend == "cpp":
            from headsup import native

            self._cpp = native.module()
            self._cfg = native.engine_config(game=self.game)
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

    # -- DREAM / ESCHER (C++ trajectory samplers) ----------------------------------------------
    def _cpp_models(self, *weight_dicts):
        from headsup import native

        return [native.make_model(w) for w in weight_dicts]

    def _fan_out(self, fn, n, seed, *args):
        chunks = self._split(n)
        seeds = np.random.SeedSequence(seed).generate_state(len(chunks), dtype=np.uint64)
        futs = [self._pool.submit(fn, k, int(s), *args) for k, s in zip(chunks, seeds)]
        return [f.result() for f in futs]

    def collect_dream(self, weights, baseline_weights, traverser, n_traversals, t, epsilon, seed):
        """DREAM outcome-sampling traversals; returns (adv Samples, value Samples (history rows,
        t = action index, target[a] = expected-SARSA target), nodes)."""
        if self.backend != "cpp":
            raise NotImplementedError("DREAM / ESCHER traversals need the C++ extension")
        nets = self._cpp_models(weights[0], weights[1], baseline_weights)
        outs = self._fan_out(lambda k, s: self._cpp.run_dream(nets[0], nets[1], nets[2], traverser, k, float(t), float(epsilon), s, self._cfg),
                             n_traversals, seed)
        return (Samples.concat([Samples(o[0], o[1], o[2]) for o in outs]), Samples.concat([Samples(o[3], o[4], o[5]) for o in outs]),
                sum(o[6] for o in outs))

    def collect_escher_values(self, weights, n_trajectories, seed):
        """ESCHER value trajectories under the current strategies: (value Samples, nodes)."""
        if self.backend != "cpp":
            raise NotImplementedError("DREAM / ESCHER traversals need the C++ extension")
        nets = self._cpp_models(weights[0], weights[1])
        outs = self._fan_out(lambda k, s: self._cpp.run_escher_values(nets[0], nets[1], k, s, self._cfg), n_trajectories, seed)
        return Samples.concat([Samples(o[0], o[1], o[2]) for o in outs]), sum(o[3] for o in outs)

    def collect_escher_regrets(self, weights, value_weights, traverser, n_trajectories, t, seed):
        """ESCHER regret trajectories: (adv Samples, strat Samples, history Samples, nodes)."""
        if self.backend != "cpp":
            raise NotImplementedError("DREAM / ESCHER traversals need the C++ extension")
        nets = self._cpp_models(weights[0], weights[1], value_weights)
        outs = self._fan_out(lambda k, s: self._cpp.run_escher_regrets(nets[0], nets[1], nets[2], traverser, k, float(t), s, self._cfg),
                             n_trajectories, seed)
        return (Samples.concat([Samples(o[0], o[1], o[2]) for o in outs]), Samples.concat([Samples(o[3], o[4], o[5]) for o in outs]),
                Samples.concat([Samples(o[6], o[7], o[8]) for o in outs]), sum(o[9] for o in outs))
