# Deep (Predictive) Discounted CFR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add VR-DeepDCFR+ and VR-DeepPDCFR+ (Xu et al. 2025) for Kuhn and Leduc, faithful to the authors' code, and reproduce the paper's exploitability curves.

**Architecture:** A new module `headsup/algos/pdcfr.py` holds a compiled game tree (`Tree`), a vectorised outcome-sampling collector (`sample_episodes`) and the solver (`PDCFRSolver`) with its CLI. It reuses the fit helpers of `headsup/algos/deep.py` (one small generalisation of `_optimise`), the reservoir buffer and exact exploitability. Tabular DCFR+ / PDCFR+ go into `headsup/algos/tabular.py` as references.

**Tech Stack:** Python 3.12, numpy, PyTorch (CPU, or CUDA graphs via the existing `_optimise`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md`

## Global Constraints

- Python is `/home/mario/workdir/headsup-poker/.venv/bin/python`; run tests CPU-only: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest ...`.
- Defaults are the authors' configs: `K = 10 000` episodes per player per iteration, `eps = 0.6`, DCFR+ `alpha 2, gamma 2, c 1.5`, PDCFR+ `alpha 2.3, gamma 2, c 1`, Adam lr `1e-3`, no gradient clipping, `R` / `r` fit 750 x 2 048, `Q` fit 1 000 x 2 048 with a target network synchronised every 50 steps, `Pi` fit 5 000 x 2 048, strategy reservoir and baseline buffer 1 000 000, networks 3 x 64 ReLU with zero output layer, utilities divided by the game's largest absolute utility.
- Exploitability unit: `exploitability()` of `headsup/algos/best_response.py` = NashConv / 2 in antes; x 1000 = mA/g.
- `DeepSolver` in `headsup/algos/deep.py` is not changed (only the helper `_optimise` gains optional arguments).
- Commit messages end with exactly `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and never contain a session link.
- Never use `pkill -f` / `pgrep -f`. Do not start runs longer than a few minutes before Task 8.
- Each new test file must run in well under a minute on CPU.

## Review Focus

1. **Fewer samples than the minibatch, or none**: a tiny budget (`traversals=5`) gives a player fewer than 2 048 advantage samples; fits must run (sampling with replacement) and an empty batch must skip the fit. Test: Task 5, `test_tiny_budget_and_untrained_solver`.
2. **Evaluation before any training**: the untrained solver's average policy must be uniform (the zero output layer under a legal-masked softmax), exploitability 0.4583 on Kuhn. Test: Task 5, same test.
3. **One-hot strategies**: at iteration 1 every network outputs zero and the fallback plays one action with probability 1; the sampler must never pick a zero-probability action (it divides by the sampling probability). Test: Task 4, `test_sampler_never_takes_a_zero_probability_action`.
4. **Resuming with other settings**: a checkpoint of one variant loaded into a solver of the other must be refused with a clear error, not half-loaded. Test: Task 6, `test_checkpoint_resume_and_variant_mismatch`.
5. **GPU fits**: the target-network copy between CUDA-graph replays and the optimisers kept across iterations only run on a GPU, which CI lacks. Test: Task 5, `test_runs_on_cuda` (skipped without CUDA; run it once locally with `CUDA_VISIBLE_DEVICES=0`).

---

### Task 1: Tabular DCFR+ and PDCFR+ references

**Files:**
- Modify: `headsup/algos/tabular.py` (class `CFR`: `__init__`, `_sigma`, `_update`; module docstring)
- Test: `tests/test_games_algos.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `CFR(game, "dcfr+" | "pdcfr+", alpha=None, gamma=None)`; `alpha` / `gamma` default per variant (`dcfr`: 1.5 / 2, `dcfr+`: 1.5 / 4, `pdcfr+`: 2.3 / 5).

- [ ] **Step 1: Write the failing test** (append to `tests/test_games_algos.py`)

```python
@pytest.mark.parametrize("variant", ["dcfr+", "pdcfr+"])
def test_discounted_plus_variants_converge(variant):
    """DCFR+ / PDCFR+ (Xu et al. 2024, arXiv 2404.13891): discounted regrets floored at zero, predictive strategy."""
    kuhn = make_game("kuhn")
    avg = CFR(kuhn, variant).iterate(200).average_policy()
    assert exploitability(kuhn, avg)[0] < 0.01
    assert expected_value(kuhn, avg) == pytest.approx(-1 / 18, abs=0.01)
    leduc = make_game("leduc")
    assert exploitability(leduc, CFR(leduc, variant).iterate(100).average_policy())[0] < 0.05  # CFR+ reaches 0.013 here
    assert CFR(kuhn, "dcfr").gamma == 2.0 and CFR(kuhn, "dcfr+").gamma == 4.0 and CFR(kuhn, "pdcfr+").alpha == 2.3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_games_algos.py -q -k discounted_plus`
Expected: FAIL with `AssertionError` from the variant check in `CFR.__init__`.

- [ ] **Step 3: Implement** in `headsup/algos/tabular.py`

Replace `CFR.__init__` and `CFR._sigma` with:

```python
_DISCOUNTS = {"dcfr": (1.5, 2.0), "dcfr+": (1.5, 4.0), "pdcfr+": (2.3, 5.0)}  # default (alpha, gamma) per variant


class CFR(_Tables):
    def __init__(self, game, variant="cfr+", alpha=None, beta=0.0, gamma=None):
        super().__init__(game)
        assert variant in ("vanilla", "lcfr", "cfr+", "dcfr", "pcfr+", "dcfr+", "pdcfr+")
        self.variant = variant
        a, g = _DISCOUNTS.get(variant, (1.5, 2.0))
        self.alpha, self.beta, self.gamma = (a if alpha is None else alpha), beta, (g if gamma is None else gamma)
        self._legal = {}

    def _plus_discount(self, t):
        """(P)DCFR+ regret discount d_t = (t-1)^alpha / ((t-1)^alpha + 1); d_1 = 0."""
        x = float(t - 1) ** self.alpha
        return x / (x + 1.0)

    def _sigma(self, key, legal):
        r = self._get(self.regret, key)
        if self.variant == "pdcfr+":  # the predicted next regret: [R d + the last instantaneous regret]^+
            last = self.last_regret.get(key)
            if last is not None:
                return regret_matching(np.maximum(r * self._plus_discount(self.iteration) + last, 0.0), legal)
            return regret_matching(r, legal)
        pred = self.last_regret.get(key) if self.variant == "pcfr+" else None
        return regret_matching(r, legal, pred)
```

In `_update`, replace the regret branch and the strategy-sum block with:

```python
    def _update(self, inst, own):
        t = self.iteration
        plus = self.variant in ("dcfr+", "pdcfr+")
        for key, r in inst.items():
            R = self._get(self.regret, key)
            if self.variant in ("vanilla", "dcfr"):
                R += r
            elif self.variant == "lcfr":
                R += t * r
            elif plus:  # discount the floored regrets, add, floor at zero
                np.maximum(R * self._plus_discount(t) + r, 0.0, out=R)
                self.last_regret[key] = r
            else:  # cfr+ / pcfr+: regret floor at zero once per iteration
                np.maximum(R + r, 0.0, out=R)
                self.last_regret[key] = r
        if plus:  # X_t = X_{t-1} ((t-1)/t)^gamma + reach * sigma_t
            decay = ((t - 1) / t) ** self.gamma
            for key, (reach_p, sigma) in own.items():
                S = self._get(self.strategy_sum, key)
                S *= decay
                S += reach_p * sigma
            return
        weight = {"vanilla": 1.0, "dcfr": 1.0, "lcfr": t, "cfr+": t, "pcfr+": t * t}[self.variant]
        for key, (reach_p, sigma) in own.items():
            self._get(self.strategy_sum, key)[:] += weight * reach_p * sigma
```

Add to the module docstring's variant list: ``` ``dcfr+`` / ``pdcfr+`` (Xu et al. 2024: discounted regrets floored at zero, PDCFR+ plays the predicted regret; alpha 1.5 / 2.3, gamma 4 / 5)```.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_games_algos.py -q`
Expected: all pass (the existing variants are unchanged: their defaults still resolve to alpha 1.5, gamma 2).

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/tabular.py tests/test_games_algos.py
git commit -m "Tabular DCFR+ and PDCFR+ reference solvers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `_optimise` steps several networks and calls a periodic hook

**Files:**
- Modify: `headsup/algos/deep.py` (function `_optimise` only)
- Test: `tests/test_deep_algos.py`

**Interfaces:**
- Produces: `_optimise(model, opt, loss_fn, steps, grad_clip=1.0, sync_every=0, sync_fn=None)` where `model` / `opt` may be lists (disjoint parameters, one loss); `sync_fn()` is called after steps 1, `1 + sync_every`, `1 + 2 * sync_every`, ... Existing calls are unaffected.

- [ ] **Step 1: Write the failing test** (append to `tests/test_deep_algos.py`)

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimise_steps_several_networks_and_syncs(device):
    import torch

    from headsup.algos.deep import _adam, _optimise

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    a, b = torch.nn.Linear(3, 1).to(device), torch.nn.Linear(3, 1).to(device)
    x = torch.randn(64, 3, device=device)
    ya, yb = x.sum(1, keepdim=True), -x.sum(1, keepdim=True)
    calls = []
    loss = _optimise([a, b], [_adam(a, 3e-2), _adam(b, 3e-2)], lambda: (a(x) - ya).pow(2).mean() + (b(x) - yb).pow(2).mean(),
                     300, grad_clip=0, sync_every=50, sync_fn=lambda: calls.append(1))
    assert loss < 0.05  # both regressions were optimised
    assert len(calls) == 6  # after steps 1, 51, 101, 151, 201, 251
```

- [ ] **Step 2: Run it to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_deep_algos.py -q -k optimise_steps`
Expected: FAIL with `TypeError: _optimise() got an unexpected keyword argument 'sync_every'`.

- [ ] **Step 3: Implement** - replace `_optimise` in `headsup/algos/deep.py` with:

```python
def _optimise(model, opt, loss_fn, steps, grad_clip=1.0, sync_every=0, sync_fn=None):
    """``steps`` optimiser steps on ``loss_fn()`` (which samples its own minibatch on the model's
    device), gradient-norm clipping as the papers.  On CUDA the whole step - sampling, forward,
    backward, clipping, fused Adam - is captured once in a CUDA graph and replayed: the small games'
    networks are launch-bound (5 ms -> ~0.3-0.7 ms per step).  ``model`` / ``opt`` may be lists (several
    networks with disjoint parameters stepped on one loss); ``sync_fn`` is called after step 1 and then
    after every ``sync_every``-th step (a target-network copy: in-place, so the graph sees it).
    Returns the last loss."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    opts = list(opt) if isinstance(opt, (list, tuple)) else [opt]
    params = [p for m in models for p in m.parameters()]
    cuda = params[0].is_cuda

    def step():
        loss = loss_fn()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip, foreach=cuda)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=not cuda)  # the graph needs the gradient buffers to stay in place
        return loss

    def after(done):
        if sync_every and (done - 1) % sync_every == 0:
            sync_fn()

    for m in models:
        m.train()
    loss = None
    if not cuda or steps <= 4:
        for i in range(steps):
            loss = step()
            after(i + 1)
    else:
        side = _side_stream(params[0].device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up outside the graph (allocates the gradient / optimiser state)
            for i in range(3):
                step()
                after(i + 1)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # recorded, not executed
            loss = step()
        for i in range(steps - 3):
            graph.replay()
            after(i + 4)
    for m in models:
        m.eval()
    return float(loss.item()) if loss is not None else float("nan")
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_deep_algos.py -q`
Expected: all pass (the CUDA parametrisation is skipped). Then once with the GPU: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/test_deep_algos.py -q -k "optimise_steps or graph or cuda"` - expected: pass.

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/deep.py tests/test_deep_algos.py
git commit -m "Small-game fits: several networks per step and a periodic hook (target-network copies)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Foundations - discount, strategies, networks, the compiled tree

**Files:**
- Create: `headsup/algos/pdcfr.py`
- Create: `tests/test_pdcfr.py`

**Interfaces:**
- Produces:
  - `discount(t: int, alpha: float, offset: float) -> float`
  - `strategy_rows(R, r, legal, d, variant, fallback) -> np.ndarray (N, A)`; `R`, `r` float `(N, A)` (`r` ignored for `"dcfr+"`), `legal` bool `(N, A)`, `variant in ("dcfr+", "pdcfr+")`, `fallback in ("authors", "argmax", "uniform")`
  - `mlp(in_dim, out_dim, hidden=64, layers=3) -> torch.nn.Sequential`
  - `Tree(game)` with arrays `kind, player, info, dec, util, child, legal, hist_x, info_obs, info_legal, info_player, chance_child, chance_prob, chance_cum`, lists `info_keys`, ints `num_nodes, num_infosets, num_decisions, num_actions, depth`, float `max_utility`, and methods `values(sigma) -> (num_nodes,)` (player 0) and `advantages(sigma, p) -> (num_infosets, A)`
  - constants `DECISION, CHANCE, TERMINAL = 0, 1, 2`, `VARIANTS = ("dcfr+", "pdcfr+")`

- [ ] **Step 1: Write the failing tests** - create `tests/test_pdcfr.py`

```python
"""Deep (Predictive) Discounted CFR (Xu et al. 2025) on Kuhn / Leduc: pieces and end-to-end convergence."""

import numpy as np
import pytest
import torch

from headsup.algos.best_response import expected_value, exploitability
from headsup.algos.pdcfr import CHANCE, DECISION, TERMINAL, Tree, discount, mlp, strategy_rows
from headsup.games import UniformPolicy, make_game


def test_discount_schedule():
    assert discount(1, 2.0, 1.5) == 0.0 and discount(1, 2.3, 1.0) == 0.0
    assert discount(2, 2.0, 1.5) == pytest.approx(1 / 2.5)
    assert discount(3, 2.3, 1.0) == pytest.approx(2**2.3 / (2**2.3 + 1))
    assert discount(500, 2.0, 1.5) == pytest.approx(1.0, abs=1e-5)


def test_strategy_rows_variants_and_fallbacks():
    legal = np.array([[True, True, True], [False, True, True], [False, True, True], [False, True, True]])
    R = np.array([[1.0, 3.0, -2.0], [9.0, -1.0, -3.0], [0.0, -1.0, -3.0], [5.0, 2.0, -4.0]])
    r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -2.0, -1.0], [0.0, -3.0, 1.0]])
    s = strategy_rows(R, r, legal, 0.5, "dcfr+", "authors")
    np.testing.assert_allclose(s[0], [0.25, 0.75, 0.0])  # regret matching on the positive part
    np.testing.assert_allclose(s[1], [0.0, 1.0, 0.0])  # nothing positive among the legal: the largest raw output
    np.testing.assert_allclose(s[3], [0.0, 1.0, 0.0])  # the illegal action's output is ignored
    p = strategy_rows(R, r, legal, 0.5, "pdcfr+", "authors")
    np.testing.assert_allclose(p[0], [0.25, 0.75, 0.0])  # max(R, 0) d + r, clipped
    np.testing.assert_allclose(p[2], [0.0, 1.0, 0.0])  # all clipped to zero: the authors' code plays the first legal action
    np.testing.assert_allclose(p[3], [0.0, 0.0, 1.0])  # prediction: 2 * 0.5 - 3 < 0 and 0 + 1 > 0
    np.testing.assert_allclose(strategy_rows(R, r, legal, 0.5, "pdcfr+", "argmax")[2], [0.0, 0.0, 1.0])  # largest unclipped prediction
    np.testing.assert_allclose(strategy_rows(R, r, legal, 0.5, "pdcfr+", "uniform")[2], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(strategy_rows(R, r, legal, 0.5, "dcfr+", "uniform")[1], [0.0, 0.5, 0.5])


def test_mlp_starts_at_zero_with_truncated_normal_hidden_layers():
    torch.manual_seed(0)
    net = mlp(34, 3)
    assert [m.out_features for m in net if isinstance(m, torch.nn.Linear)] == [64, 64, 64, 3]
    assert float(net(torch.randn(5, 34)).abs().max()) == 0.0
    w = net[0].weight
    assert float(w.abs().max()) <= 2 / 34**0.5 + 1e-6 and float(net[0].bias.abs().max()) == 0.0


@pytest.mark.parametrize("name,nodes,infosets,terminals,depth,scale", [("kuhn", 58, 12, 30, 3, 2.0), ("leduc", 9457, 936, 5520, 8, 13.0)])
def test_tree_matches_the_game(name, nodes, infosets, terminals, depth, scale):
    g = make_game(name)
    t = Tree(g)
    assert (t.num_nodes, t.num_infosets, int((t.kind == TERMINAL).sum()), t.depth, t.max_utility) == (nodes, infosets, terminals, depth, scale)
    assert t.hist_x.shape == (t.num_decisions, 2 * g.obs_dim) and t.info_obs.shape == (infosets, g.obs_dim)
    dec = t.kind == DECISION
    assert (t.dec[dec] >= 0).all() and (t.dec[~dec] == -1).all() and (t.info[dec] >= 0).all()
    np.testing.assert_array_equal(t.legal[dec], t.info_legal[t.info[dec]])  # an infoset's histories share its legal actions
    np.testing.assert_array_equal(t.player[dec], t.info_player[t.info[dec]])
    assert (t.child[t.legal] > np.nonzero(t.legal)[0]).all()  # children come after their parents
    ch = t.kind == CHANCE
    np.testing.assert_allclose(t.chance_prob[ch].sum(1), 1.0)
    # exact values: the uniform profile's value of the game equals the reference implementation's
    sigma = t.info_legal / t.info_legal.sum(1, keepdims=True)
    assert t.values(sigma)[0] == pytest.approx(expected_value(g, UniformPolicy(g)), abs=1e-12)


def test_tree_advantages_are_counterfactual_gains():
    """A player's advantage-weighted strategy is zero, and a best response has no positive advantage left."""
    from headsup.algos.tabular import CFR

    g = make_game("kuhn")
    t = Tree(g)
    rng = np.random.default_rng(0)
    sigma = np.where(t.info_legal, rng.random(t.info_legal.shape), 0.0)
    sigma /= sigma.sum(1, keepdims=True)
    for p in (0, 1):
        adv = t.advantages(sigma, p)
        mine = t.info_player == p
        np.testing.assert_allclose((adv * sigma).sum(1)[mine], 0.0, atol=1e-12)
        assert not adv[~mine].any() and not adv[~t.info_legal].any()
    eq = CFR(g, "cfr+").iterate(2000).average_policy()
    table = np.stack([eq.table[k] for k in t.info_keys])
    assert max(t.advantages(table, 0).max(), t.advantages(table, 1).max()) < 0.02  # (near-)equilibrium: nothing to gain
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'headsup.algos.pdcfr'`.

- [ ] **Step 3: Implement** - create `headsup/algos/pdcfr.py`

```python
"""Deep (Predictive) Discounted CFR on the game protocol: VR-DeepDCFR+ and VR-DeepPDCFR+.

Xu, Li, Fu, Fu, Xing & Cheng 2025 (arXiv 2511.08174), following the authors' code (rpSebastian/DeepPDCFR):

* per player one persistent network ``R`` of cumulative *advantages*, fitted every iteration to its own previous
  output - clipped at zero and discounted by ``d_t = (t-1)^alpha / ((t-1)^alpha + c)`` - plus the advantages
  sampled in this iteration only (no reservoir, no iteration weights); the clip is applied where ``R`` is read;
* VR-DeepPDCFR+ adds a network ``r`` of the latest instantaneous advantages and plays regret matching on the
  predicted next regret ``max(R, 0) d_t + r``;
* outcome sampling (the traverser explores with ``eps``), variance-reduced by a history-action baseline ``Q``
  (player 0's value; negated for player 1) that is re-fitted from scratch with expected-SARSA targets;
* the result is an average-strategy network fitted on a reservoir of ``(I, t, sigma_t(I))`` with weights
  ``(t / T)^gamma``.

Where the paper and the code disagree the defaults follow the code (see docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md).
Small games only: the whole tree is compiled to arrays and the episodes of an iteration are sampled in one
vectorised pass.

    python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 10000000 --json runs/x.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from headsup.algos.best_response import TabularPolicy, exploitability
from headsup.algos.deep import _adam, _batch_index, _optimise
from headsup.deepcfr.memory import ReservoirBuffer

VARIANTS = ("dcfr+", "pdcfr+")
# the authors' configs: alpha, gamma and the discount denominator's constant (the paper's formula has 1 for both)
DEFAULTS = {"dcfr+": dict(alpha=2.0, gamma=2.0, offset=1.5), "pdcfr+": dict(alpha=2.3, gamma=2.0, offset=1.0)}
DECISION, CHANCE, TERMINAL = 0, 1, 2


def discount(t, alpha, offset):
    """d_t = (t-1)^alpha / ((t-1)^alpha + offset): how much of the clipped cumulative advantage survives into
    iteration t (0 at t = 1)."""
    x = float(t - 1) ** alpha
    return x / (x + offset)


def strategy_rows(R, r, legal, d, variant, fallback="authors"):
    """Current strategies for rows of network outputs: regret matching on ``R`` (dcfr+) or on the predicted
    regret ``max(R, 0) d + r`` (pdcfr+), over the legal actions.  When nothing is positive: ``authors`` - the
    largest raw output (dcfr+) / the first legal action (pdcfr+: their code clips before the argmax);
    ``argmax`` - the largest unclipped score; ``uniform``."""
    R = np.where(legal, R, 0.0)
    score = R if variant == "dcfr+" else np.maximum(R, 0.0) * d + np.where(legal, r, 0.0)
    pos = np.maximum(score, 0.0)
    total = pos.sum(-1, keepdims=True)
    if fallback == "uniform":
        fb = legal / legal.sum(-1, keepdims=True)
    else:
        if fallback == "authors" and variant == "pdcfr+":
            best = np.argmax(legal, axis=-1)
        else:
            best = np.argmax(np.where(legal, score, -np.inf), axis=-1)
        fb = np.zeros_like(pos)
        np.put_along_axis(fb, best[..., None], 1.0, axis=-1)
    return np.where(total > 0, pos / np.maximum(total, 1e-300), fb)


def mlp(in_dim, out_dim, hidden=64, layers=3):
    """The authors' network: ReLU layers initialised from a normal truncated at two standard deviations of
    1 / sqrt(fan_in) with zero bias, and an output layer that starts at zero."""
    mods, d = [], in_dim
    for _ in range(layers):
        lin = nn.Linear(d, hidden)
        sd = 1.0 / math.sqrt(d)
        nn.init.trunc_normal_(lin.weight, std=sd, a=-2 * sd, b=2 * sd)
        nn.init.zeros_(lin.bias)
        mods += [lin, nn.ReLU()]
        d = hidden
    head = nn.Linear(d, out_dim)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*mods, head)


class Tree:
    """The whole game as arrays: nodes in depth-first order (a parent's id is smaller than its children's).
    Decision nodes carry their infoset index (``info``) and their index among the decision nodes (``dec``: a
    row of ``hist_x``, the concatenated infostates of both players - the baseline's input)."""

    def __init__(self, game):
        A = self.num_actions = game.num_actions
        kind, player, info, dec, util, child, legal, hist_x, chance = [], [], [], [], [], [], [], [], {}
        index, self.info_keys, info_obs, info_legal, info_player = {}, [], [], [], []
        self.depth = 0  # the largest number of decisions on a path

        def add(state, depth):
            i = len(kind)
            kind.append(DECISION)
            player.append(-1)
            info.append(-1)
            dec.append(-1)
            util.append(0.0)
            child.append([-1] * A)
            legal.append([False] * A)
            if state.is_terminal():
                kind[i], util[i] = TERMINAL, float(state.returns()[0])
                self.depth = max(self.depth, depth)
            elif state.is_chance():
                kind[i] = CHANCE
                outcomes = state.chance_outcomes()
                chance[i] = ([add(state.child(a), depth) for a, _ in outcomes], [p for _, p in outcomes])
            else:
                p = state.current_player
                key = state.info_key(p)
                if key not in index:
                    index[key] = len(self.info_keys)
                    self.info_keys.append(key)
                    info_obs.append(state.info_state(p))
                    info_legal.append(state.legal_mask())
                    info_player.append(p)
                player[i], info[i], dec[i] = p, index[key], len(hist_x)
                hist_x.append(np.concatenate([state.info_state(0), state.info_state(1)]))
                for a in state.legal_actions():
                    legal[i][a] = True
                    child[i][a] = add(state.child(a), depth + 1)
            return i

        add(game.new_initial_state(), 0)
        self.kind = np.array(kind, dtype=np.int8)
        self.player = np.array(player, dtype=np.int8)
        self.info = np.array(info, dtype=np.int64)
        self.dec = np.array(dec, dtype=np.int64)
        self.util = np.array(util)  # player 0's utility at the terminals
        self.child = np.array(child, dtype=np.int64)
        self.legal = np.array(legal, dtype=bool)
        self.hist_x = np.stack(hist_x).astype(np.float32)
        self.info_obs = np.stack(info_obs).astype(np.float32)
        self.info_legal = np.stack(info_legal).astype(bool)
        self.info_player = np.array(info_player)
        self.num_nodes, self.num_infosets, self.num_decisions = len(kind), len(self.info_keys), len(hist_x)
        width = max(len(kids) for kids, _ in chance.values())
        self.chance_child = np.full((self.num_nodes, width), -1, dtype=np.int64)
        self.chance_prob = np.zeros((self.num_nodes, width))
        self.chance_cum = np.full((self.num_nodes, width), 2.0)  # padding above every uniform draw
        for i, (kids, probs) in chance.items():
            self.chance_child[i, : len(kids)] = kids
            self.chance_prob[i, : len(kids)] = probs
            cum = np.cumsum(probs)
            cum[-1] = 1.0
            self.chance_cum[i, : len(kids)] = cum
        self.max_utility = float(np.abs(self.util).max())

    def values(self, sigma):
        """Player 0's expected utility at every node under the profile ``sigma`` (num_infosets, A)."""
        v = self.util.copy()
        for i in range(self.num_nodes - 1, -1, -1):
            if self.kind[i] == CHANCE:
                k = self.chance_child[i] >= 0
                v[i] = self.chance_prob[i, k] @ v[self.chance_child[i, k]]
            elif self.kind[i] == DECISION:
                a = self.legal[i]
                v[i] = sigma[self.info[i], a] @ v[self.child[i, a]]
        return v

    def advantages(self, sigma, p):
        """Exact advantages of player p under ``sigma``: for each of p's infosets I and legal action a,
        sum_h pi_-p(h) (v_p(h a) - v_p(h)) / sum_h pi_-p(h) over the histories h of I (pi_-p: the opponent's
        and chance's reach).  (num_infosets, A); zero for the other player's and for unreachable infosets."""
        v = self.values(sigma) * (1.0 if p == 0 else -1.0)
        reach = np.zeros(self.num_nodes)
        reach[0] = 1.0
        num = np.zeros((self.num_infosets, self.num_actions))
        den = np.zeros(self.num_infosets)
        for i in range(self.num_nodes):
            if self.kind[i] == CHANCE:
                k = self.chance_child[i] >= 0
                reach[self.chance_child[i, k]] = reach[i] * self.chance_prob[i, k]
            elif self.kind[i] == DECISION:
                a = self.legal[i]
                kids = self.child[i, a]
                if self.player[i] == p:
                    reach[kids] = reach[i]
                    num[self.info[i], a] += reach[i] * (v[kids] - v[i])
                    den[self.info[i]] += reach[i]
                else:
                    reach[kids] = reach[i] * sigma[self.info[i], a]
        return num / np.maximum(den, 1e-300)[:, None]
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q`
Expected: 7 passed (the two parametrised tree cases count separately).

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/pdcfr.py tests/test_pdcfr.py
git commit -m "Deep PDCFR+: discount, strategies, networks and the compiled game tree

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The vectorised outcome-sampling collector

**Files:**
- Modify: `headsup/algos/pdcfr.py` (add `sample_episodes` after `Tree`)
- Test: `tests/test_pdcfr.py`

**Interfaces:**
- Consumes: `Tree` (Task 3).
- Produces: `sample_episodes(tree, sigma, q, traverser, n, epsilon, rng, scale=1.0) -> dict` with numpy arrays
  `adv_info (m,) int64`, `adv (m, A) float64` (utilities / `scale`), `adv_reach (m,) float64` (the traverser's own sampling reach of the node), `strat_info (k,) int64`, `q_node`, `q_action`, `q_next`, `q_next_info` (all `(j,) int64`; `q_next` / `q_next_info` are 0 when done), `q_reward (j,) float64` (player 0's terminal utility / `scale`, else 0), `q_done (j,) float32`, and `nodes: int` (decision + terminal + chance nodes visited).
  `sigma`: `(num_infosets, A)`; `q`: `(num_decisions, A)` baseline of player 0's value (zeros = no baseline).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pdcfr.py`; add `sample_episodes` to the import from `headsup.algos.pdcfr`)

```python
def _random_profile(tree, rng, floor=0.25):
    s = np.where(tree.info_legal, rng.random(tree.info_legal.shape), 0.0)
    s /= s.sum(1, keepdims=True)
    return floor * tree.info_legal / tree.info_legal.sum(1, keepdims=True) + (1 - floor) * s


@pytest.mark.parametrize("baseline", [False, True])
def test_sampled_advantages_are_unbiased_with_any_baseline(baseline):
    """E[sampled advantage | infoset visited] is the exact advantage of the current strategy - with no baseline
    and with an arbitrary one (the baseline only changes the variance)."""
    g = make_game("kuhn")
    tree = Tree(g)
    rng = np.random.default_rng(1)
    sigma = _random_profile(tree, rng)
    q = rng.normal(scale=0.5, size=(tree.num_decisions, 3)) if baseline else np.zeros((tree.num_decisions, 3))
    for p in (0, 1):
        data = sample_episodes(tree, sigma, q, p, 400_000, 0.6, rng, scale=tree.max_utility)
        total = np.zeros((tree.num_infosets, 3))
        np.add.at(total, data["adv_info"], data["adv"])
        count = np.bincount(data["adv_info"], minlength=tree.num_infosets)
        mine = tree.info_player == p
        assert count[mine].min() > 5000 and not count[~mine].any()
        exact = tree.advantages(sigma, p) / tree.max_utility
        np.testing.assert_allclose(total[mine] / count[mine][:, None], exact[mine], atol=0.03)
        assert not data["adv"][~tree.info_legal[data["adv_info"]]].any()  # illegal actions carry no advantage
        assert (tree.info_player[data["strat_info"]] == 1 - p).all()


def test_sampler_transitions_and_counts():
    g = make_game("leduc")
    tree = Tree(g)
    rng = np.random.default_rng(0)
    sigma = _random_profile(tree, rng)
    n = 2000
    data = sample_episodes(tree, sigma, np.zeros((tree.num_decisions, 3)), 0, n, 0.6, rng, scale=13.0)
    j = len(data["q_node"])
    assert j == len(data["adv_info"]) + len(data["strat_info"])  # one transition per decision
    assert int(data["q_done"].sum()) == n  # every episode ends exactly once
    done = data["q_done"] > 0
    assert not data["q_reward"][~done].any() and np.abs(data["q_reward"]).max() <= 1.0
    assert (data["q_next"][done] == 0).all() and (data["q_next_info"][done] == 0).all()
    assert tree.legal[np.flatnonzero(tree.dec >= 0)[data["q_node"]], data["q_action"]].all()  # only legal actions are taken
    chance_per_episode = 3  # two private cards, one public card (when round 2 is reached)
    assert n * 2 + j + n <= data["nodes"] <= n * chance_per_episode + j + n
    assert (data["adv_reach"] > 0).all() and (data["adv_reach"] <= 1.0).all()


def test_sampler_never_takes_a_zero_probability_action():
    """One-hot strategies (iteration 1: every network outputs zero) and no exploration: every sampled action has
    probability one, values stay finite."""
    g = make_game("leduc")
    tree = Tree(g)
    last = tree.num_actions - 1 - np.argmax(tree.info_legal[:, ::-1], axis=1)  # the last legal action of each infoset
    sigma = np.zeros(tree.info_legal.shape)
    sigma[np.arange(tree.num_infosets), last] = 1.0
    data = sample_episodes(tree, sigma, np.zeros((tree.num_decisions, 3)), 1, 20_000, 0.0, np.random.default_rng(3), scale=13.0)
    node = np.flatnonzero(tree.dec >= 0)[data["q_node"]]
    assert (data["q_action"] == last[tree.info[node]]).all()
    assert np.isfinite(data["adv"]).all() and np.abs(data["adv"]).max() <= 2.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q -k "sampl"`
Expected: collection error `ImportError: cannot import name 'sample_episodes'`.

- [ ] **Step 3: Implement** - add to `headsup/algos/pdcfr.py` after the `Tree` class

```python
def sample_episodes(tree, sigma, q, traverser, n, epsilon, rng, scale=1.0):
    """``n`` outcome-sampling episodes for ``traverser``, all at once.  The traverser samples
    ``xi = eps * uniform + (1 - eps) * sigma``, the opponent ``sigma``, chance its distribution.  Backwards along
    each trajectory, at every decision node h with sampled action a* and child h':

        v(h, a) = Q_i(h, a) + [a = a*] (v(h') - Q_i(h, a)) / xi(a*),      v(h) = sum_a sigma(a) v(h, a)

    with Q_i = +-q (the baseline of player 0's value) and terminal values u_i / scale.  Traverser nodes give an
    advantage sample v(h, .) - v(h); opponent nodes a strategy sample; every decision a baseline transition."""
    sign = 1.0 if traverser == 0 else -1.0
    visited = 0

    def deal(nodes):  # resolve chance nodes (several can follow each other)
        nonlocal visited
        while True:
            c = np.flatnonzero(tree.kind[nodes] == CHANCE)
            if len(c) == 0:
                return nodes
            j = (rng.random(len(c))[:, None] >= tree.chance_cum[nodes[c]]).sum(1)
            nodes[c] = tree.chance_child[nodes[c], j]
            visited += len(c)

    rows = np.arange(n)  # the episodes still running
    h = deal(np.zeros(n, dtype=np.int64))  # their current decision nodes
    value = np.zeros(n)  # per episode: the value of the node below the step being processed
    reach = np.ones(n)  # the traverser's own sampling reach of the current node
    steps = []
    while len(rows):
        visited += len(rows)
        info, legal = tree.info[h], tree.legal[h]
        sig = sigma[info]
        mine = tree.player[h] == traverser
        xi = np.where(mine[:, None], epsilon * legal / legal.sum(1, keepdims=True) + (1.0 - epsilon) * sig, sig)
        cum = np.cumsum(xi, axis=1)
        # inverse CDF: the number of cumulative sums not above the draw; an action with probability 0 is never hit
        a = (rng.random(len(rows))[:, None] * cum[:, -1:] >= cum).sum(1)
        ar = np.arange(len(rows))
        p_a = xi[ar, a] / cum[:, -1]
        nxt = deal(tree.child[h, a])
        done = tree.kind[nxt] == TERMINAL
        visited += int(done.sum())
        value[rows[done]] = sign * tree.util[nxt[done]] / scale
        steps.append(dict(rows=rows, h=h, info=info, legal=legal, sig=sig, mine=mine, a=a, p=p_a, nxt=nxt, done=done, reach=reach[rows]))
        reach[rows[mine]] *= p_a[mine]
        rows, h = rows[~done], nxt[~done]

    adv_info, adv, adv_reach, strat_info = [], [], [], []
    for s in reversed(steps):
        ar = np.arange(len(s["rows"]))
        qh = sign * q[tree.dec[s["h"]]] * s["legal"]
        qh[ar, s["a"]] += (value[s["rows"]] - qh[ar, s["a"]]) / s["p"]
        v = (qh * s["sig"]).sum(1)
        mine = s["mine"]
        adv_info.append(s["info"][mine])
        adv.append(((qh - v[:, None]) * s["legal"])[mine])
        adv_reach.append(s["reach"][mine])
        strat_info.append(s["info"][~mine])
        value[s["rows"]] = v
    cat = np.concatenate
    return {
        "adv_info": cat(adv_info), "adv": cat(adv), "adv_reach": cat(adv_reach), "strat_info": cat(strat_info),
        "q_node": cat([tree.dec[s["h"]] for s in steps]),
        "q_action": cat([s["a"] for s in steps]),
        "q_next": cat([np.where(s["done"], 0, tree.dec[s["nxt"]]) for s in steps]),
        "q_next_info": cat([np.where(s["done"], 0, tree.info[s["nxt"]]) for s in steps]),
        "q_reward": cat([np.where(s["done"], tree.util[s["nxt"]] / scale, 0.0) for s in steps]),
        "q_done": cat([s["done"].astype(np.float32) for s in steps]),
        "nodes": visited,
    }
```

Note `reach[rows]` is a copy (fancy indexing), taken before the traverser's own action is multiplied in.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q`
Expected: all pass. If `test_sampled_advantages_are_unbiased_with_any_baseline` fails only by a margin slightly above `atol` for the random baseline, raise the episode count to 1 000 000 (it runs in about a second) - do not loosen the tolerance beyond 0.03.

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/pdcfr.py tests/test_pdcfr.py
git commit -m "Deep PDCFR+: vectorised outcome-sampling collector with baseline-corrected values

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The solver - fits, iteration, evaluation

**Files:**
- Modify: `headsup/algos/pdcfr.py` (add `Transitions`, `advantage_target`, `baseline_target`, `PDCFRSolver`)
- Test: `tests/test_pdcfr.py`

**Interfaces:**
- Consumes: `Tree`, `sample_episodes`, `strategy_rows`, `discount`, `mlp` (Tasks 3-4); `_optimise(..., sync_every, sync_fn)` (Task 2); `_adam`, `_batch_index` from `headsup/algos/deep.py`; `ReservoirBuffer(capacity, device, obs_dim, target_dim, seed, int_dim=0, legal_dim=A)` with `.add(obs, t, target, legal)`, fields `.obs_float`, `.t`, `.target`, `.legal`, `len()`.
- Produces:
  - `advantage_target(frozen_out, legal, d, adv) -> Tensor`: `clamp(frozen_out * legal, min=0) * d + adv`
  - `baseline_target(reward, done, next_q, next_sigma) -> Tensor`: `reward + (1 - done) * (next_q * next_sigma).sum(1)`
  - `PDCFRSolver(game, variant="pdcfr+", traversals=10_000, epsilon=0.6, alpha=None, gamma=None, discount_offset=None, adv_steps=750, adv_batch=2048, q_steps=1000, q_batch=2048, q_capacity=1_000_000, q_sync=50, policy_steps=5000, policy_batch=2048, strat_capacity=1_000_000, lr=1e-3, hidden=64, layers=3, fallback="authors", reinit_prediction=False, baseline=True, reach_weighted=False, device="cpu", seed=0)` with attributes `iteration`, `episodes`, `nodes_touched`, `tree`, `sigma`, `q_tab` and methods `iterate(n=1)`, `current_policy()`, `policy_net()`, `average_policy()`, `evaluate() -> {"current": float, "average": float}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pdcfr.py`; extend the import with `PDCFRSolver, advantage_target, baseline_target`)

```python
def test_advantage_target_clips_the_previous_output_at_read_time():
    frozen = torch.tensor([[2.0, -3.0, 5.0]])
    legal = torch.tensor([[1.0, 1.0, 0.0]])
    adv = torch.tensor([[0.5, -0.25, 0.0]])
    torch.testing.assert_close(advantage_target(frozen, legal, 0.5, adv), torch.tensor([[1.5, -0.25, 0.0]]))
    torch.testing.assert_close(advantage_target(frozen, legal, 0.0, adv), adv)  # iteration 1: d = 0


def test_baseline_target_is_expected_sarsa():
    reward = torch.tensor([0.0, -1.0])
    done = torch.tensor([0.0, 1.0])
    next_q = torch.tensor([[1.0, 3.0, 100.0], [7.0, 7.0, 7.0]])
    next_sigma = torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(baseline_target(reward, done, next_q, next_sigma), torch.tensor([2.5, -1.0]))


def _small(variant, **kw):
    args = dict(traversals=1000, adv_steps=100, adv_batch=256, q_steps=100, q_batch=256, policy_steps=400, policy_batch=256, seed=0)
    args.update(kw)
    return PDCFRSolver(make_game("kuhn"), variant, **args)


@pytest.mark.parametrize("variant", ["dcfr+", "pdcfr+"])
def test_kuhn_converges(variant):
    s = _small(variant)
    assert (s.alpha, s.gamma, s.offset) == ((2.0, 2.0, 1.5) if variant == "dcfr+" else (2.3, 2.0, 1.0))
    s.iterate(15)
    ev = s.evaluate()
    assert ev["average"] < 0.15 and np.isfinite(ev["current"]), ev  # uniform: 0.458
    assert s.iteration == 15 and s.episodes == 15 * 2 * 1000 and s.nodes_touched > s.episodes * 3
    pol = s.average_policy()
    for probs in pol.table.values():
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert (s.r is None) == (variant == "dcfr+")


def test_ablation_switches_run():
    for kw in (dict(baseline=False), dict(reach_weighted=True), dict(reinit_prediction=True), dict(fallback="uniform")):
        s = _small("pdcfr+", traversals=300, **kw).iterate(3)
        assert np.isfinite(s.evaluate()["average"])
    assert not _small("pdcfr+", traversals=300, baseline=False).iterate(2).q_tab.any()


def test_tiny_budget_and_untrained_solver():
    g = make_game("kuhn")
    s = PDCFRSolver(g, "pdcfr+", traversals=5, adv_steps=5, q_steps=5, policy_steps=5, seed=0)
    assert s.evaluate()["average"] == pytest.approx(exploitability(g, UniformPolicy(g))[0], abs=1e-9)  # untrained: uniform
    s.iterate(3)  # far fewer samples than the 2 048 minibatch
    assert np.isfinite(s.evaluate()["average"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda():
    s = _small("pdcfr+", traversals=500, device="cuda").iterate(3)
    assert np.isfinite(s.evaluate()["average"]) and np.isfinite(s.q_tab).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q -k "target or kuhn or ablation or tiny"`
Expected: collection error `ImportError: cannot import name 'PDCFRSolver'`.

- [ ] **Step 3: Implement** - add to `headsup/algos/pdcfr.py` after `sample_episodes`

```python
class Transitions:
    """Circular buffer of baseline transitions, stored as decision-node / infoset indices."""

    FIELDS = (("node", np.int64), ("action", np.int64), ("next", np.int64), ("next_info", np.int64),
              ("reward", np.float32), ("done", np.float32))

    def __init__(self, capacity):
        self.capacity, self.pos, self.size = int(capacity), 0, 0
        self.data = {k: np.zeros(self.capacity, dtype=t) for k, t in self.FIELDS}

    def add(self, **cols):
        n = len(cols["node"])
        idx = (self.pos + np.arange(n)) % self.capacity
        for k, _ in self.FIELDS:
            self.data[k][idx] = cols[k]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)


def advantage_target(frozen_out, legal, d, adv):
    """The regression target of the cumulative-advantage net: the previous net's output clipped at zero (the
    "+" of DCFR+, applied where the net is read), discounted, plus this iteration's sampled advantages."""
    return torch.clamp(frozen_out * legal, min=0.0) * d + adv


def baseline_target(reward, done, next_q, next_sigma):
    """Expected SARSA for the baseline: the reward plus the next state's values under the next strategy."""
    return reward + (1.0 - done) * (next_q * next_sigma).sum(1)


class PDCFRSolver:
    def __init__(self, game, variant="pdcfr+", traversals=10_000, epsilon=0.6, alpha=None, gamma=None, discount_offset=None,
                 adv_steps=750, adv_batch=2048, q_steps=1000, q_batch=2048, q_capacity=1_000_000, q_sync=50,
                 policy_steps=5000, policy_batch=2048, strat_capacity=1_000_000, lr=1e-3, hidden=64, layers=3,
                 fallback="authors", reinit_prediction=False, baseline=True, reach_weighted=False, device="cpu", seed=0):
        assert variant in VARIANTS and fallback in ("authors", "argmax", "uniform")
        self.game, self.variant = game, variant
        d = DEFAULTS[variant]
        self.alpha = d["alpha"] if alpha is None else alpha
        self.gamma = d["gamma"] if gamma is None else gamma
        self.offset = d["offset"] if discount_offset is None else discount_offset
        self.traversals, self.epsilon, self.lr = traversals, epsilon, lr
        self.adv_steps, self.adv_batch = adv_steps, adv_batch
        self.q_steps, self.q_batch, self.q_sync = q_steps, q_batch, q_sync
        self.policy_steps, self.policy_batch = policy_steps, policy_batch
        self.hidden, self.layers = hidden, layers
        self.fallback, self.reinit_prediction = fallback, reinit_prediction
        self.baseline, self.reach_weighted = baseline, reach_weighted
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.tree = tree = Tree(game)
        self.scale = tree.max_utility  # utilities are normalised to [-1, 1]
        A, D = game.num_actions, game.obs_dim
        self.info_obs_t = torch.as_tensor(tree.info_obs, device=self.device)
        self.info_legal_t = torch.as_tensor(tree.info_legal, dtype=torch.float32, device=self.device)
        self.hist_x_t = torch.as_tensor(tree.hist_x, device=self.device)
        # persistent networks and optimisers: they are never re-initialised (unless ``reinit_prediction``)
        self.R = [self._net(D) for _ in range(2)]
        self.opt_R = [_adam(n, lr) for n in self.R]
        self.r = [self._net(D) for _ in range(2)] if variant == "pdcfr+" else None
        self.opt_r = [_adam(n, lr) for n in self.r] if self.r else None
        self.q_state = None  # the baseline net of the last fit (re-created at every fit), for checkpoints
        self.q_tab = np.zeros((tree.num_decisions, A))  # player 0's baseline values at every decision node
        self.q_memory = Transitions(q_capacity)
        self.strat_memory = ReservoirBuffer(strat_capacity, self.device, obs_dim=D, target_dim=A, seed=seed + 2, int_dim=0, legal_dim=A)
        self.iteration = self.episodes = self.nodes_touched = 0
        self.sigma = self._strategy_table(0.0)

    def _net(self, in_dim):
        return mlp(in_dim, self.game.num_actions, self.hidden, self.layers).to(self.device).eval()

    def _strategy_table(self, d):
        """Both players' strategies at every infoset from the current networks, with discount ``d``."""
        with torch.no_grad():
            R = torch.stack([n(self.info_obs_t) for n in self.R]).cpu().numpy().astype(np.float64)
            r = torch.stack([n(self.info_obs_t) for n in self.r]).cpu().numpy().astype(np.float64) if self.r else R
        own = self.tree.info_player
        rows = np.arange(len(own))
        return strategy_rows(R[own, rows], r[own, rows], self.tree.info_legal, d, self.variant, self.fallback)

    # -- fits ----------------------------------------------------------------------------------------
    def _fit_advantage(self, p, data, d):
        m = len(data["adv_info"])
        if m == 0:
            return
        idx = torch.as_tensor(data["adv_info"], device=self.device)
        adv = data["adv"] / data["adv_reach"][:, None] if self.reach_weighted else data["adv"]  # the paper's "w/o adv" ablation
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        legal, obs, R = self.info_legal_t, self.info_obs_t, self.R[p]
        with torch.no_grad():  # the network as it is before this fit, at every infoset
            frozen = R(obs)
        nets, opts = [R], [self.opt_R[p]]
        r = None
        if self.r:
            if self.reinit_prediction:  # the paper's table; the authors' code never re-initialises it
                self.r[p] = self._net(self.game.obs_dim)
                self.opt_r[p] = _adam(self.r[p], self.lr)
            r = self.r[p]
            nets.append(r)
            opts.append(self.opt_r[p])

        def loss_fn():
            j = _batch_index(m, self.adv_batch, self.device)
            i = idx[j]
            x, lg = obs[i], legal[i]
            loss = (R(x) * lg - advantage_target(frozen[i], lg, d, adv[j])).pow(2).mean()
            if r is not None:  # the same minibatch
                loss = loss + (r(x) * lg - adv[j]).pow(2).mean()
            return loss

        _optimise(nets, opts, loss_fn, self.adv_steps, grad_clip=0)

    def _fit_baseline(self, t):
        mem = self.q_memory
        n = mem.size
        if n == 0:
            return
        q, target = self._net(2 * self.game.obs_dim), self._net(2 * self.game.obs_dim)
        target.load_state_dict(q.state_dict())
        # the strategies the next iteration will play, from the networks just updated
        sigma_next = torch.as_tensor(self._strategy_table(discount(t + 1, self.alpha, self.offset)), dtype=torch.float32, device=self.device)
        col = {k: torch.as_tensor(v[:n], device=self.device) for k, v in mem.data.items()}
        hist = self.hist_x_t

        def loss_fn():
            j = _batch_index(n, self.q_batch, self.device)
            pred = q(hist[col["node"][j]]).gather(1, col["action"][j][:, None]).squeeze(1)
            with torch.no_grad():
                tgt = baseline_target(col["reward"][j], col["done"][j], target(hist[col["next"][j]]), sigma_next[col["next_info"][j]])
            return (pred - tgt).pow(2).mean()

        _optimise(q, _adam(q, self.lr), loss_fn, self.q_steps, grad_clip=0, sync_every=self.q_sync,
                  sync_fn=lambda: target.load_state_dict(q.state_dict()))
        with torch.no_grad():
            self.q_tab = q(hist).cpu().numpy().astype(np.float64)
        self.q_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}

    # -- the iteration ---------------------------------------------------------------------------------
    def iterate(self, n=1):
        tree = self.tree
        for _ in range(n):
            self.iteration += 1
            t = self.iteration
            d = discount(t, self.alpha, self.offset)
            self.sigma = self._strategy_table(d)
            for p in (0, 1):  # alternating: player 1's episodes already use player 0's updated networks
                data = sample_episodes(tree, self.sigma, self.q_tab, p, self.traversals, self.epsilon, self.rng, self.scale)
                self.episodes += self.traversals
                self.nodes_touched += data["nodes"]
                k = data["strat_info"]
                if len(k):
                    self.strat_memory.add(tree.info_obs[k], np.full(len(k), t, np.float32), self.sigma[k], tree.info_legal[k])
                if self.baseline:
                    self.q_memory.add(node=data["q_node"], action=data["q_action"], next=data["q_next"],
                                      next_info=data["q_next_info"], reward=data["q_reward"], done=data["q_done"])
                self._fit_advantage(p, data, d)
                self.sigma = self._strategy_table(d)
                if self.baseline:
                    self._fit_baseline(t)
        return self

    # -- policies and evaluation ---------------------------------------------------------------------------
    def _tabular(self, probs):
        return TabularPolicy(self.game, {k: probs[i] for i, k in enumerate(self.tree.info_keys)})

    def current_policy(self):
        """The strategy the next iteration would play."""
        return self._tabular(self._strategy_table(discount(self.iteration + 1, self.alpha, self.offset)))

    def policy_net(self):
        """The average-strategy network, fitted from scratch on the strategy reservoir with weights (2 t / T)^gamma."""
        net = self._net(self.game.obs_dim)
        buf = self.strat_memory
        n = len(buf)
        if n == 0:
            return net
        scale, gamma = 2.0 / max(self.iteration, 1), self.gamma

        def loss_fn():
            i = _batch_index(n, self.policy_batch, self.device)
            logits = net(buf.obs_float[i])
            probs = torch.softmax(torch.where(buf.legal[i] > 0, logits, torch.full_like(logits, -1e20)), dim=-1)
            w = (buf.t[i] * scale).pow(gamma)
            return (w[:, None] * (probs - buf.target[i]).pow(2)).mean()

        _optimise(net, _adam(net, self.lr), loss_fn, self.policy_steps, grad_clip=0)
        return net

    def average_policy(self):
        net = self.policy_net()
        with torch.no_grad():
            logits = net(self.info_obs_t)
            probs = torch.softmax(torch.where(self.info_legal_t > 0, logits, torch.full_like(logits, -1e20)), dim=-1)
        return self._tabular(probs.cpu().numpy().astype(np.float64))

    def evaluate(self):
        return {"current": exploitability(self.game, self.current_policy())[0],
                "average": exploitability(self.game, self.average_policy())[0]}
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q`
Expected: all pass (`test_runs_on_cuda` skipped). Then once: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/test_pdcfr.py -q -k cuda` - expected: 1 passed.
If `test_kuhn_converges` misses the 0.15 bound, print `ev` for 15 / 30 / 60 iterations before changing anything: the exploitability must fall with the iterations. A flat or rising curve is a bug in the fits (check the sign of `q_tab` for player 1 and that `d` is applied to the frozen output), not a threshold problem.

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/pdcfr.py tests/test_pdcfr.py
git commit -m "Deep PDCFR+: the solver (bootstrapped advantage fits, baseline, average-policy network)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Checkpoints and the command line

**Files:**
- Modify: `headsup/algos/pdcfr.py` (add `state_dict` / `load_state_dict` to `PDCFRSolver`, `main`)
- Test: `tests/test_pdcfr.py`

**Interfaces:**
- Consumes: `PDCFRSolver` (Task 5); `ReservoirBuffer.state_dict()` / `.load_state_dict(state)` and its `.rng`.
- Produces: `PDCFRSolver.state_dict() -> dict`, `PDCFRSolver.load_state_dict(state) -> self` (raises `ValueError` for another variant); `main(argv=None)` with flags `--game --variant --episodes --traversals --epsilon --alpha --gamma --discount-offset --adv-steps --adv-batch --q-steps --q-batch --policy-steps --policy-batch --fallback --reinit-prediction --no-baseline --reach-weighted --eval-every --checkpoint --checkpoint-minutes --device --seed --json`. Progress line: `"{game} {variant} it {it}: exploitability current {c:.4f} average {a:.4f}  nodes {n:.3g}  episodes {e}  ({s:.0f}s)"`; JSON `{"game", "algo", "args", "curve": [{"iteration", "current", "average", "nodes_touched", "episodes", "seconds"}]}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pdcfr.py`)

```python
def test_checkpoint_resume_and_variant_mismatch(tmp_path):
    a = _small("pdcfr+", traversals=300).iterate(2)
    torch.save(a.state_dict(), tmp_path / "ck.pt")
    state = torch.load(tmp_path / "ck.pt", weights_only=False)
    b = _small("pdcfr+", traversals=300).load_state_dict(state)
    assert (b.iteration, b.episodes, b.nodes_touched) == (a.iteration, a.episodes, a.nodes_touched)
    np.testing.assert_array_equal(b.sigma, a.sigma)
    np.testing.assert_array_equal(b.q_tab, a.q_tab)
    assert b.q_memory.size == a.q_memory.size and len(b.strat_memory) == len(a.strat_memory)
    torch.set_rng_state(state["torch_rng"])  # both solvers share torch's global generator in this process
    a.iterate(1)
    torch.set_rng_state(state["torch_rng"])
    b.iterate(1)  # the same random streams: the resumed run continues identically (CPU)
    np.testing.assert_allclose(b.sigma, a.sigma, atol=1e-6)
    with pytest.raises(ValueError, match="variant"):
        _small("dcfr+", traversals=300).load_state_dict(a.state_dict())


def test_cli_writes_a_curve_and_resumes(tmp_path, capsys):
    import json

    from headsup.algos.pdcfr import main

    args = ["--game", "kuhn", "--variant", "dcfr+", "--traversals", "200", "--adv-steps", "20", "--q-steps", "20", "--policy-steps", "50",
            "--checkpoint", str(tmp_path / "ck.pt"), "--checkpoint-minutes", "0", "--json", str(tmp_path / "run.json")]
    main(args + ["--episodes", "1600"])  # 4 iterations
    out = capsys.readouterr().out
    assert "kuhn dcfr+ it 4: exploitability current" in out and "episodes 1600" in out
    curve = json.load(open(tmp_path / "run.json"))["curve"]
    assert [c["iteration"] for c in curve] == [1, 2, 3, 4] and curve[-1]["episodes"] == 1600  # 1, 2, every 3rd, the last
    main(args + ["--episodes", "2400"])  # resumes at iteration 4 and runs to 6
    assert "resumed from" in capsys.readouterr().out
    assert [c["iteration"] for c in json.load(open(tmp_path / "run.json"))["curve"]] == [1, 2, 3, 4, 6]
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q -k "checkpoint or cli"`
Expected: FAIL with `AttributeError: 'PDCFRSolver' object has no attribute 'state_dict'` and `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement** - add these methods to `PDCFRSolver` and `main` at the end of `headsup/algos/pdcfr.py`

```python
    # -- checkpoints -----------------------------------------------------------------------------------
    def state_dict(self):
        cpu = lambda net: {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}  # noqa: E731
        mem = self.q_memory
        return {
            "variant": self.variant, "iteration": self.iteration, "episodes": self.episodes, "nodes_touched": self.nodes_touched,
            "R": [cpu(n) for n in self.R], "opt_R": [o.state_dict() for o in self.opt_R],
            "r": [cpu(n) for n in self.r] if self.r else None, "opt_r": [o.state_dict() for o in self.opt_r] if self.r else None,
            "q_state": self.q_state, "q_tab": self.q_tab.copy(), "sigma": self.sigma.copy(),
            "q_memory": {"pos": mem.pos, "size": mem.size, **{k: v[: mem.size].copy() for k, v in mem.data.items()}},
            "strat_memory": self.strat_memory.state_dict(), "strat_rng": self.strat_memory.rng.bit_generator.state,
            "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        if state["variant"] != self.variant:
            raise ValueError(f"checkpoint of variant {state['variant']!r} cannot be loaded into a {self.variant!r} solver")
        self.iteration, self.episodes, self.nodes_touched = int(state["iteration"]), int(state["episodes"]), int(state["nodes_touched"])
        for net, opt, sd, osd in zip(self.R, self.opt_R, state["R"], state["opt_R"]):
            net.load_state_dict(sd)
            opt.load_state_dict(osd)
        if self.r:
            for net, opt, sd, osd in zip(self.r, self.opt_r, state["r"], state["opt_r"]):
                net.load_state_dict(sd)
                opt.load_state_dict(osd)
        self.q_state, self.q_tab, self.sigma = state["q_state"], state["q_tab"].copy(), state["sigma"].copy()
        mem, saved = self.q_memory, state["q_memory"]
        mem.pos, mem.size = int(saved["pos"]), int(saved["size"])
        for k, _ in Transitions.FIELDS:
            mem.data[k][: mem.size] = saved[k]
        self.strat_memory.load_state_dict(state["strat_memory"])
        self.strat_memory.rng.bit_generator.state = state["strat_rng"]
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        return self


def main(argv=None):
    from headsup.games import make_game

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--variant", default="pdcfr+", choices=VARIANTS, help="dcfr+ = VR-DeepDCFR+, pdcfr+ = VR-DeepPDCFR+")
    p.add_argument("--episodes", type=int, default=10_000_000, help="sampled episodes in total; iterations = episodes // (2 x traversals)")
    p.add_argument("--traversals", type=int, default=10_000, help="episodes per player per iteration")
    p.add_argument("--epsilon", type=float, default=0.6, help="the traverser's exploration")
    p.add_argument("--alpha", type=float, default=None, help="regret discount exponent (default: 2 for dcfr+, 2.3 for pdcfr+)")
    p.add_argument("--gamma", type=float, default=None, help="average-strategy weight exponent (default 2)")
    p.add_argument("--discount-offset", type=float, default=None,
                   help="constant in the discount's denominator (default: the authors' code - 1.5 for dcfr+, 1 for pdcfr+; the paper: 1)")
    p.add_argument("--adv-steps", type=int, default=750)
    p.add_argument("--adv-batch", type=int, default=2048)
    p.add_argument("--q-steps", type=int, default=1000, help="baseline fit steps (the authors' configs; the paper's table says 10000)")
    p.add_argument("--q-batch", type=int, default=2048)
    p.add_argument("--policy-steps", type=int, default=5000)
    p.add_argument("--policy-batch", type=int, default=2048)
    p.add_argument("--fallback", default="authors", choices=["authors", "argmax", "uniform"],
                   help="strategy when no (predicted) regret is positive: the authors' code, the largest unclipped score, or uniform")
    p.add_argument("--reinit-prediction", action="store_true", help="re-initialise the prediction net every iteration (the paper's table)")
    p.add_argument("--no-baseline", action="store_true", help="no variance reduction (the paper's DeepPDCFR+ ablation)")
    p.add_argument("--reach-weighted", action="store_true", help="divide the samples by the traverser's sampling reach (the 'w/o adv' ablation)")
    p.add_argument("--eval-every", type=int, default=3, help="fit and evaluate the average policy every this many iterations (and at 1, 2 and the last)")
    p.add_argument("--checkpoint", default=None, help="saved at evaluations; an existing file is resumed from")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0, help="at most one checkpoint per this many minutes (plus the final one)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    game = make_game(args.game)
    solver = PDCFRSolver(game, args.variant, traversals=args.traversals, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma,
                         discount_offset=args.discount_offset, adv_steps=args.adv_steps, adv_batch=args.adv_batch, q_steps=args.q_steps,
                         q_batch=args.q_batch, policy_steps=args.policy_steps, policy_batch=args.policy_batch, fallback=args.fallback,
                         reinit_prediction=args.reinit_prediction, baseline=not args.no_baseline, reach_weighted=args.reach_weighted,
                         device=args.device, seed=args.seed)
    iterations = args.episodes // (2 * args.traversals)
    curve, elapsed = [], 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        solver.load_state_dict(saved["solver"])
        curve, elapsed = saved["curve"], saved["seconds"]
        print(f"resumed from {args.checkpoint} at iteration {solver.iteration}", flush=True)
    t0 = time.perf_counter() - elapsed
    last_save = time.perf_counter()

    def dump():
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"game": args.game, "algo": args.variant, "args": vars(args), "curve": curve}, f, indent=2)

    for it in range(solver.iteration + 1, iterations + 1):
        solver.iterate()
        if it % args.eval_every == 0 or it < args.eval_every or it == iterations:
            ev = solver.evaluate()
            curve.append({"iteration": it, **ev, "nodes_touched": solver.nodes_touched, "episodes": solver.episodes,
                          "seconds": time.perf_counter() - t0})
            print(f"{args.game} {args.variant} it {it}: exploitability current {ev['current']:.4f} average {ev['average']:.4f}  "
                  f"nodes {solver.nodes_touched:.3g}  episodes {solver.episodes}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            dump()
            due = time.perf_counter() - last_save >= 60.0 * args.checkpoint_minutes or it == iterations
            if args.checkpoint and due:  # atomic: a crash while writing leaves the previous checkpoint intact
                torch.save({"solver": solver.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}, args.checkpoint + ".tmp")
                os.replace(args.checkpoint + ".tmp", args.checkpoint)
                last_save = time.perf_counter()
    dump()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_pdcfr.py -q`
Expected: all pass. Then the whole suite once: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q -x -p no:cacheprovider` - expected: all pass (about 5 minutes).

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/pdcfr.py tests/test_pdcfr.py
git commit -m "Deep PDCFR+: checkpoints and the command line

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The report table against the paper's curves

**Files:**
- Modify: `headsup/algos/leduc_report.py` (`load_run`, `_at` callers, `table`, `main`, constants)
- Test: `tests/test_deep_algos.py`

**Interfaces:**
- Consumes: the JSON curves of Task 6 (rows with `"episodes"`).
- Produces: setups `"pdcfrp"` (Leduc) and `"pdcfrk"` (Kuhn) for `--row setup,algo,label,glob` with `algo in ("dcfr+", "pdcfr+")`; `PDCFR_PAPER[(setup, algo)] = {episodes: mA/g}`; `load_run` rows gain a 4th element (episodes, `None` when absent).

- [ ] **Step 1: Write the failing test** (append to `tests/test_deep_algos.py`)

```python
def test_leduc_report_tabulates_pdcfr_runs_by_episodes(tmp_path):
    import json

    from headsup.algos.leduc_report import PDCFR_PAPER, load_run, table

    curve = [{"iteration": i, "average": 0.2 - 0.0002 * i, "current": 1.0, "nodes_touched": 150_000 * i, "episodes": 20_000 * i}
             for i in range(1, 501)]
    (tmp_path / "x_s0.json").write_text(json.dumps({"curve": curve}))
    assert load_run(str(tmp_path / "x_s0.json"))[49] == (50, pytest.approx(190.0), 7_500_000, 1_000_000)
    out = table([("pdcfrp", "pdcfr+", "ours", str(tmp_path / "x_s*.json"))], "pdcfrp", (1e6, 4e6, 9.5e6))
    lines = out.splitlines()
    assert lines[0].startswith("| episodes |") and "9.5e6" in lines[0]
    assert lines[2] == "| **VR-DeepPDCFR+, paper** | **158** | **115** | **90** |"
    assert lines[3].startswith("| ours (1) | 190 | 160 | 105 |")  # means within +-10 % (the last column: 9-10 M) of x
    assert PDCFR_PAPER[("pdcfrk", "dcfr+")][9.5e6] == pytest.approx(5.3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_deep_algos.py -q -k pdcfr_runs`
Expected: FAIL with `ImportError: cannot import name 'PDCFR_PAPER'`.

- [ ] **Step 3: Implement** in `headsup/algos/leduc_report.py`

Add after `NODE_RATIO`:

```python
# Xu et al. (2025), Fig. 1 (4-seed means read off the PDF's vector curves; x 1000 = mA/g): episodes -> value; the
# last point is the mean over 9-10 M episodes
PDCFR_PAPER = {("pdcfrp", "dcfr+"): {1e6: 152, 2e6: 121, 4e6: 114, 8e6: 85, 9.5e6: 89},
               ("pdcfrp", "pdcfr+"): {1e6: 158, 2e6: 121, 4e6: 115, 8e6: 88, 9.5e6: 90},
               ("pdcfrk", "dcfr+"): {1e6: 8.7, 2e6: 7.0, 4e6: 5.2, 8e6: 5.6, 9.5e6: 5.3},
               ("pdcfrk", "pdcfr+"): {1e6: 4.1, 2e6: 4.3, 4e6: 4.1, 8e6: 4.3, 9.5e6: 3.3}}
_NAMES = {"sdcfr": "SD-CFR", "deepcfr": "Deep CFR", "dream": "DREAM", "dcfr+": "VR-DeepDCFR+", "pdcfr+": "VR-DeepPDCFR+"}
```

Append to `DEFAULT_ROWS` (the globs of Task 8's runs):

```python
    ("pdcfrp", "dcfr+", "VR-DeepDCFR+, ours", "runs/leduc_pdcfr/leduc_dcfr+_s*.json"),
    ("pdcfrp", "pdcfr+", "VR-DeepPDCFR+, ours", "runs/leduc_pdcfr/leduc_pdcfr+_s*.json"),
    ("pdcfrk", "dcfr+", "VR-DeepDCFR+, ours", "runs/leduc_pdcfr/kuhn_dcfr+_s*.json"),
    ("pdcfrk", "pdcfr+", "VR-DeepPDCFR+, ours", "runs/leduc_pdcfr/kuhn_pdcfr+_s*.json"),
```

Replace `load_run` with:

```python
def load_run(path):
    """[(iteration, average mA/g, nodes touched, episodes or None)] of one run (a --json curve or a log)."""
    if path.endswith(".json"):
        return [(c["iteration"], 1000 * c["average"], c["nodes_touched"], c.get("episodes")) for c in json.load(open(path))["curve"]]
    rows = {int(i): (int(i), 1000 * float(a), float(n), None) for i, _, a, n in _LINE.findall(open(path).read())}
    return [rows[k] for k in sorted(rows)]
```

Replace `table` with:

```python
def table(rows, setup, xs):
    """Markdown table: one line per row of ``setup`` and per algorithm the paper's line."""
    episodes = setup in ("pdcfrp", "pdcfrk")
    head = "iteration" if setup == "sdcfrp" else "episodes" if episodes else "nodes touched (DREAM code's count)"
    out = [f"| {head} | " + " | ".join(f"{x:g}" if setup == "sdcfrp" else f"{x:.2g}".replace("e+0", "e") for x in xs) + " |",
           "|---|" + "---|" * len(xs)]
    digits = 1 if setup == "pdcfrk" else 0
    done = set()
    for st, algo, label, pattern in rows:
        if st != setup:
            continue
        runs = [load_run(p) for p in sorted(glob.glob(pattern))]
        runs = [r for r in runs if r]
        if setup == "sdcfrp":
            cells = [_fmt([_at(r, 0, x) for r in runs]) for x in xs]
        elif episodes:  # the final column (9.5e6) averages the evaluations between 9 M and 10 M episodes
            cells = [_fmt([_at(r, 3, x, window=0.0527 if x == 9.5e6 else 0.1) for r in runs], digits) for x in xs]
        else:
            if algo not in NODE_RATIO:
                raise ValueError(f"no DREAM-code node count ratio for {algo!r} (known: {sorted(NODE_RATIO)})")
            cells = [_fmt([_at(r, 2, x, NODE_RATIO[algo]) for r in runs]) for x in xs]
        out.append(f"| {label} ({len(runs)}) | " + " | ".join(cells) + " |")
        if algo not in done:
            done.add(algo)
            if setup == "sdcfrp":
                col = 0 if algo == "sdcfr" else 1
                paper = [_interp({k: v[col] for k, v in SDCFR_PAPER.items()}, x) for x in xs]
            elif episodes:
                paper = [PDCFR_PAPER.get((setup, algo), {}).get(x) for x in xs]
            else:
                paper = [_interp(DREAM_PAPER[algo], x) if algo in DREAM_PAPER else None for x in xs]
            name = "ES-SD-CFR" if (algo, setup) == ("sdcfr", "dreamp") else _NAMES.get(algo, algo)
            out.insert(len(out) - 1, f"| **{name}, paper** | " + " | ".join("–" if v is None else f"**{v:.{digits}f}**" for v in paper) + " |")
    return "\n".join(out)
```

Change `_fmt` to take the number of decimals:

```python
def _fmt(vals, digits=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.{digits}f}" + (f" ± {np.std(vals, ddof=1):.{digits}f}" if len(vals) > 1 else "")
```

In `main`, after the DREAM table, add:

```python
    xs = (1e6, 2e6, 4e6, 8e6, 9.5e6)
    print("\nDeep PDCFR+ paper setup (10 000 episodes per player and iteration), mA/g by episodes (last column: mean over 9-10 M):\n")
    print("Leduc\n")
    print(table(rows, "pdcfrp", xs))
    print("\nKuhn\n")
    print(table(rows, "pdcfrk", xs))
```

and extend the module docstring's sentence on setups with: ``` ``pdcfrp`` / ``pdcfrk`` = the Deep PDCFR+ paper's Leduc / Kuhn setup (x-axis: episodes)```.

`_at(rows, key, x, scale=1.0, window=0.1)` already takes `window`; rows without episodes (`None`) must not break it - in `_at` replace the comprehension with:

```python
    vals = [r[1] for r in rows if r[key] is not None and (1 - window) * x <= r[key] / scale <= (1 + window) * x]
```

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_deep_algos.py -q -k "report"` and `.venv/bin/python -m headsup.algos.leduc_report | tail -15`
Expected: tests pass; the report prints the two new tables with the paper rows and `(0)` runs / dashes for ours (no runs yet), and the existing tables are unchanged.

- [ ] **Step 5: Commit**

```bash
git add headsup/algos/leduc_report.py tests/test_deep_algos.py
git commit -m "Leduc report: Deep PDCFR+ tables by episodes against the paper's curves

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Validation runs, results, documentation

**Files:**
- Create: `runs/scripts/pdcfr.sh` (git-ignored, like the other run scripts)
- Modify: `README.md` (Algorithms list, Results section), `CLAUDE.md` (repository map line), `docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md` (record the outcome)

**Interfaces:**
- Consumes: the CLI of Task 6, the report of Task 7.
- Produces: `runs/leduc_pdcfr/{kuhn,leduc}_{dcfr+,pdcfr+}_s{0,1,2}.json`.

- [ ] **Step 1: Time one iteration on each device and pick the device**

```bash
cd /home/mario/workdir/headsup-poker
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 .venv/bin/python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 120000 --device cpu | tail -3
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 120000 --device cuda:0 | tail -3
```

Expected: 6 iterations each; note the seconds at iteration 6. Use the faster device for the runs (a full run is 500 iterations; expect well under 3 hours). Record both timings in the commit message of Step 5.

- [ ] **Step 2: Launch the twelve runs** - create `runs/scripts/pdcfr.sh`

```bash
#!/bin/bash
# Deep PDCFR+ paper setup (Xu et al. 2025): Kuhn and Leduc, VR-DeepDCFR+ / VR-DeepPDCFR+, 3 seeds, 10 M episodes.
# usage: pdcfr.sh <device: cpu | cuda:0>
cd /home/mario/workdir/headsup-poker || exit 1
source .venv/bin/activate
DEV=${1:-cpu}
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES="" || export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
D=runs/leduc_pdcfr; mkdir -p $D
run() {  # game variant seed: resumes from its checkpoint after a crash
  name=$1_$2_s$3; attempt=0
  while true; do attempt=$((attempt + 1))
    python -m headsup.algos.pdcfr --game $1 --variant $2 --seed $3 --episodes 10000000 --device $DEV \
      --json $D/$name.json --checkpoint $D/$name.ck.pt >> $D/$name.log 2>&1 && return 0
    echo "$(date) $name crashed (attempt $attempt); resuming in 2 min" >> $D/$name.log; sleep 120
  done
}
for G in kuhn leduc; do for V in dcfr+ pdcfr+; do for S in 0 1 2; do run $G $V $S & sleep 5; done; done; done
wait
echo "$(date) pdcfr runs done" >> runs/queue.log
```

Run: `chmod +x runs/scripts/pdcfr.sh && (nohup runs/scripts/pdcfr.sh <device> > /dev/null 2>&1 &)` and watch with `runs/scripts/watch_retry.sh 'runs/leduc_pdcfr/*.log' 120 30 20` (extend its `live()` pattern with `$3=="headsup.algos.pdcfr"` first). With `cuda:0`, twelve processes cost about 12 x 6 GB of host memory under WSL: launch the six Kuhn runs first and the six Leduc runs when they finish, or use `cpu`.

- [ ] **Step 3: Compare with the success criteria**

Run: `.venv/bin/python -m headsup.algos.leduc_report | tail -16`
Expected (spec, "Success criteria"): Leduc 3-seed means in the last column inside 51-133 (VR-DeepDCFR+) and 63-147 (VR-DeepPDCFR+); Kuhn within a factor of 2 of 5.3 / 3.3. If a criterion fails, do not tune: first compare the curve shape at 1 M / 2 M / 4 M / 8 M with the paper's row, then test the two documented deviations one at a time on one seed (`--q-steps 10000`; `--fallback argmax`), and report what was found.

- [ ] **Step 4: Document**

README, "Algorithms" list - add a line in the style of its neighbours:

```markdown
- **Deep PDCFR+** (`python -m headsup.algos.pdcfr`): VR-DeepDCFR+ / VR-DeepPDCFR+ (Xu et al. 2025) on Kuhn / Leduc - persistent advantage networks bootstrapped from their own discounted, clipped output; outcome sampling with a learned baseline.
```

README, "Leduc: reproductions" - add the two tables printed by the report (paper rows and ours, with the measured numbers), the command

```bash
python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 10000000 --device cuda
```

and one sentence stating the outcome against the paper's 95 % bands and that the defaults follow the authors' code where it differs from the paper (discount constant 1.5 for DCFR+, 1 000 baseline steps, no re-initialisation of the prediction net, first-legal-action fallback). Add the paper to "References". CLAUDE.md: add `headsup/algos/pdcfr.py` to the repository description where the small-game solvers are listed. Spec: append a "Result" section with the final numbers and the measured run time.

- [ ] **Step 5: Commit and push**

```bash
git add README.md CLAUDE.md docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md
git commit -m "Deep PDCFR+: Kuhn / Leduc results against the paper

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```
