# Neural Fictitious Self-Play on the small games - design

Status: written and implemented autonomously on 2026-10-08 (the work was pre-approved); the three-seed validation runs
to 3e6 iterations are still to be done (first curves: "Success criteria" below).
Sources: Heinrich & Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games",
arXiv 1603.01121 (2016); OpenSpiel @ 48401890 (`open_spiel/python/pytorch/nfsp.py`, `dqn.py`); the DREAM authors'
NFSP (github.com/EricSteinberger/DREAM: `NFSP/`, `Leduc_NFSP.py`, `PokerRL/rl/agent_modules/DDQN.py`); Steinberger,
Lerer & Brown, "DREAM", arXiv 2006.10410, Fig. 2.

## Goal

Add NFSP for Kuhn and Leduc as a self-contained solver, `headsup/algos/nfsp.py`, with two presets - the paper's Leduc
setup and the DREAM authors' NFSP as their Leduc script configures it - and reproduce both published Leduc curves.

Non-goals: hold'em (FHP / HULH need a two-seat vectorised engine), the paper's LHE experiment, greedy-average or
best-response play as the evaluated strategy.

## Algorithm

Per player `i`: an action-value network `Q_i` with a target copy, an average-policy network `Pi_i`, a circular memory
`M_RL` and a reservoir `M_SL`. `N` tables (128) are played in parallel; all of them always wait at a decision node.

One iteration `t` = `steps` environment steps (128: one decision at each table), then `updates` (2) SGD steps per
network:

1. **Play.** At every table the player to act, in information state `s` with legal actions `L(s)`:
   - *best-response mode*: `a = argmax_{a in L(s)} Q_i(s, a)`, or uniform over `L(s)` with probability `eps_t`;
     `(s, a)` goes to `M_SL` (exploratory actions included);
   - *average mode*: `a ~ softmax(Pi_i(s))` over `L(s)`;
   - the player's previous decision at this table, if any, is completed: `(s_prev, a_prev, 0, s, not done)` goes to
     `M_RL` - in both modes.
   The action is applied and chance is sampled. When the hand ends, both players' last decisions are completed with
   `(s_prev, a_prev, +-u / scale, -, done)`, the table is dealt again and **each seat draws its mode for the new
   hand**: best response with probability `eta`, else average. The mode is held until the hand ends.
2. **Q updates** (each player, `updates` times, once `M_RL` holds a minibatch): a minibatch of 128 transitions drawn
   with replacement; loss `mean (Q(s, a) - y)^2` with `y = r + (1 - done) max_{a' in L(s')} Q'(s', a')` (no discount).
   Double DQN (`dream`): `y = r + (1 - done) Q'(s', argmax_{a' in L(s')} Q(s', a'))`. After every `target_every`
   (300) updates of `Q_i`: `Q'_i <- Q_i`.
3. **Pi updates** (likewise, once `M_SL` holds a minibatch): loss `mean -log softmax_{L(s)}(Pi(s))[a]`.

The result is the profile of the two average-policy networks; its exploitability is computed exactly
(`headsup.algos.best_response.exploitability`, antes, mean over the two seats).

Exploration: `eps_t = eps_0 / sqrt(t)` (`paper`) or `eps_0 / (1 + c sqrt(t - 1))` (`dream`, `c = 0.01`), `t` = the
iteration (1, 2, ...).

Illegal actions never enter: the argmax, the max over next actions and the policy logits are restricted to `L(s)`.

## Presets

| | `--preset paper` (Heinrich & Silver, Leduc) | `--preset dream` (`Leduc_NFSP.py`) |
|---|---|---|
| networks | MLP, 1 hidden layer x 64 ReLU, PyTorch default initialisation | `game.make_model(arch="deepcfr_dueling")`: Deep-CFR body (64), dueling `Q` head, `Pi` head |
| optimiser | plain SGD, lr 0.1 (`Q`) / 0.005 (`Pi`) | plain SGD, lr 0.1 / 0.01, gradient-norm clip 1.0 |
| RL loss | DQN, squared TD | Double DQN, squared TD |
| exploration | `0.06 / sqrt(t)`, one coin per decision | `0.06 / (1 + 0.01 sqrt(t - 1))`, one coin per seat and step for all tables |
| reward unit | utilities / 2.6 (`scale` = largest utility / 5; Kuhn: 0.4) - see "The reward unit" | the same (the DREAM code: stack / 5) |
| minibatch, updates | 128, 2 per network and iteration | same |
| iteration | 128 steps (128 tables x 1) | same |
| `M_RL` | 200 000, circular | same |
| `M_SL` | 2 000 000, reservoir | same (no minimum insertion probability) |
| target refit | every 300 `Q` updates | same |
| `eta` | 0.1 | same |

Every row is a keyword of `NFSPSolver` and a command-line option; a preset only sets the defaults.

## Where the sources disagree

| detail | paper | OpenSpiel | DREAM code | `paper` | `dream` | switch |
|---|---|---|---|---|---|---|
| network | MLP 1 x 64 | MLP [128] | Deep-CFR-style body, dueling head | paper | DREAM (the repository's version of that net, see below) | `--arch`, `--hidden`, `--layers` |
| learning rates `Q` / `Pi` | 0.1 / 0.005 | 0.01 / 0.01 | 0.1 / 0.01 | paper | DREAM | `--lr-q`, `--lr-pi` |
| gradient clipping | none stated | optional, off | norm 1.0 | none | 1.0 | `--grad-clip` |
| RL target | DQN | DQN | Double DQN | DQN | Double DQN | `--double-dqn` / `--no-double-dqn` |
| reward unit | not stated | antes | stack / 5 | DREAM (calibrated, see below) | DREAM | `--reward-scale` (1: antes) |
| `eps` schedule | 0.06, "proportionally to the inverse square root of the number of iterations" | 0.06 -> 0.001 "exp" over 2e7 inner steps (ends near 0.023) | `0.06 / (1 + 0.01 sqrt(iter))` | `0.06 / sqrt(t)` | DREAM | `--eps-start`, `--eps-const` |
| exploration coin | per decision | per decision | one for the whole batch of a seat's tables | per decision | DREAM | `--shared-explore` / `--no-shared-explore` |
| what "128 steps" counts | not defined | each agent's own steps (1 update per 64) | steps of both players (2 updates per 128) | DREAM's reading | DREAM | `--steps` (256 = OpenSpiel's ratio), `--updates` |
| parallel tables | not stated | 1 | 128 x 1 step | 128 | 128 | `--envs` |
| learning starts | not stated | 1000 samples | a minibatch (128) | a minibatch | a minibatch | - |
| `M_SL` row | `(s, a)` | `(s, eps-greedy probabilities)` | `(s, a)` | `(s, a)` | `(s, a)` | - |
| `M_SL` insertion | reservoir | reservoir | reservoir with `max(capacity / seen, min_prob)`, `min_prob` 0 | reservoir | reservoir | `--sl-min-prob`, `--sl-window` |
| target refit | 300 updates | counted in best-response-mode steps of the inner DQN (about every 3000 updates at `eta` 0.1) | 300 `Q` updates | 300 `Q` updates | 300 `Q` updates | `--target-every` |
| minibatch draw | not stated | without replacement | with replacement | with | with | - |
| legal masking | not in Algorithm 1 | argmax, max, logits | argmax, max, logits | all three | all three | - |
| reported number | "exploitability" (inferred: antes, mean over seats) | NashConv (2 x) | mA/g, mean over seats | antes, mean over seats | same | - |

The paper never defines an iteration. As in the research brief it is taken to be 128 steps followed by the updates:
this is what the DREAM code does, and a 2M sliding window then fills after about 3e5 iterations, where the paper's
sliding-window curve starts to diverge.

## The reward unit of the `paper` preset

The paper gives the learning rate (0.1, plain SGD) but not the unit of the rewards, and the two are one setting: the
size of a step is the product. With the natural reading - antes, Leduc utilities up to 13 - the preset does not
reproduce the paper. Measured on Leduc, exploitability in mA/g (seeds 0 / 1 / 2, the evaluation at that iteration; the
paper's curve for comparison):

| iterations | 1e5 | 2e5 | 3e5 | 5e5 | 6e5 | 8e5 | 1e6 |
|---|---|---|---|---|---|---|---|
| paper, Fig. 1a (64 units) | 430 | 257 | 213 | 158 | 150 | 138 | 128 |
| rewards in antes (`--reward-scale 1`) | 264 / 307 / 305 | 275 / 211 / 297 | 234 / 155 / 298 | 275 / 188 / 225 | 283 / 220 / 239 | 304 / - / - | - |
| utilities / 2.6 (the preset) | 291 / 367 / 268 | 146 / 178 / 295 | 120 / 130 / 303 | 95 / 116 / 202 | 87 / 96 / 166 | 91 / 90 / 151 | 86 / 87 / 146 |
| utilities / 13 | 260 / 335 / - | 179 / 256 / - | 126 / 148 / - | - | - | - | - |

With antes the curves stop falling near 250 mA/g and drift upwards; with the DREAM code's unit they keep falling and
their three-seed mean is within a factor 1.5 of the paper's curve from 1e3 to 1e6 iterations. What goes wrong with
antes, measured on checkpoints (antes: seed 0 at 490k and seed 1 at 290k / 580k iterations; scaled: seed 0 at 290k):

- 27-34 of the 64 hidden units of each `Q` network are active on no infoset at all (scaled: 17-18; with
  `--lr-q 0.02`: 4-10) and the hidden activations reach 10-15 (scaled: 5): a step of 0.1 on errors of several antes
  kills ReLU units.
- Against the exact action values of a best response to what the opponent actually plays (`Pi` with probability
  `1 - eta`, greedy `Q` with `eta`), `Q` is off by 0.62-0.78 antes (visit-weighted rmse; scaled: 0.46-0.51), and the
  greedy policy realises 12-54 % of the gain of the exact best response over `Pi` (scaled: 60 %).
- `Pi` itself is not the problem: it is within 0.01 (total variation) of the empirical average of `M_SL`. A 1 x 64
  MLP can represent the exact action values (rmse 0.02 by regression). More exploration (`--eps-const 0.01`), Double
  DQN and a smaller `Q` learning rate did not remove the plateau (one or two seeds each).

So the `paper` preset takes the unit of the DREAM code, whose authors ran NFSP at the same learning rate and report it
as the paper's hyperparameters; `--reward-scale 1` gives antes. Normalising to [-1, 1] (`--reward-scale 13`) behaves
like 2.6 as far as it was run. The choice was made on three seeds to 1e6 iterations and is the main thing the
validation runs test.

## Deliberate deviations

From all three sources:

1. **Inputs.** The repository's perfect-recall info-state features (Leduc 34, Kuhn 22), not the paper's 30-feature
   card / betting tensor, OpenSpiel's tensor or PokerRL's observation.
2. **The networks are numpy arrays with a hand-written backward pass** (as `headsup/numpy_model.py` mirrors the
   hold'em network for the traversal workers). An SGD step on a 64-unit network is bounded by per-operation overhead,
   not arithmetic: on this machine a `Q` step with torch autograd and `torch.optim.SGD` costs 480-570 us for the
   1 x 64 MLP and 3.5 ms for `game.make_model(arch="deepcfr_dueling")`, against 0.09-0.14 ms and 1.0-1.4 ms here (see
   "Compute"). The parameters keep the torch module's names and layout, the initial weights are that module's, and the
   tests require the forward pass and one whole `Q` / `Pi` update to equal the torch module's with autograd,
   `clip_grad_norm_` and `torch.optim.SGD`. `--device` exists for symmetry with the other solvers and accepts `cpu`
   only (a GPU was not measured: these networks are launch-bound there as well).
3. **The target network is stored as its outputs at every infoset** (one forward pass over the game's infosets per
   refit instead of one per minibatch) - the same numbers.
4. **Memories hold infoset indices**, not observations (the tree is compiled to arrays): `M_SL` is 5 bytes a row.
5. **The reservoir is the textbook Algorithm R** (item `m` is kept with probability `capacity / m`); the DREAM code
   divides by the number of items seen before it.
6. **A zero standard deviation in the dueling net's normalisation gives a zero gradient**, where torch gives NaN.

`paper` preset, from the paper:

7. **Rewards are divided by 2.6** (the largest utility / 5): the paper states no unit; see the section above.
8. 128 parallel tables with one step each per iteration instead of (presumably) one game played sequentially: a hand
   spans several iterations, the data are the same.
9. `Q` and `Pi` are not updated before their memory holds a minibatch (the paper does not say).
10. `eps_t = 0.06 / sqrt(t)` is the literal reading of "decayed to 0, proportionally to the inverse square root of the
    number of iterations"; the time constant is not given (the DREAM code's schedule is the same law with another
    one: `--eps-const 0.01`).

`dream` preset, from the DREAM code:

11. **Network body.** `deepcfr_dueling` is the repository's net (docs/paper-fidelity.md): a 64-wide card branch without
    skip connection, three trunk layers with the skip added after the ReLU, normalisation without gain and bias; the
    authors' has a 192-wide card branch and a history branch with skips, two trunk layers and a LayerNorm with gain and
    bias. Its `Pi` head has a hidden layer (64) before the logits; the authors' is a single linear layer.
12. **Kuhn's reward scale** is the largest utility / 5 = 0.4 by analogy (PokerRL has no Kuhn).
13. Iteration 0 of the DREAM driver skips the `Pi` updates; here they start when `M_SL` holds a minibatch (iteration
    ~20), which subsumes it.

Not copied from OpenSpiel: the inner DQN's counters that only advance in best-response-mode steps, the soft
`M_SL` targets, the minimum buffer size of 1000.

## Architecture

- `headsup/algos/nfsp.py` (new)
  - `epsilon(t, start, const)`, `td_target(reward, done, next_target, next_legal, next_online=None)`,
    `cross_entropy(logits, legal, action)` - the three formulas, as pure functions.
  - `MLP` / `Dueling`: the numpy networks (`forward`, `backward`, `step`, `state_dict`, `torch_module`).
  - `Memory`: named columns, circular or reservoir (with `min_prob`).
  - `chance_closure(tree)`: every deal as one draw (the two private cards are two chance nodes in a row).
  - `NFSPSolver(game, preset="paper", seed=0, **overrides)`: `iterate()`, `average_policy()`, `evaluate()`,
    `state_dict()` / `load_state_dict()` (the whole state: networks, target values, memories, tables, modes, pending
    transitions, counters, random generator; a snapshot shares nothing with the solver; a checkpoint written with
    other settings is refused). Plain SGD has no optimiser state.
  - CLI `python -m headsup.algos.nfsp --game leduc --preset paper --iterations 3000000 --seed 0 --json ...
    --checkpoint ...`. Evaluations at 1, 2, 5, 10, 20, 50, ... iterations up to `--eval-every` (10 000), then at its
    multiples, and at the end. The JSON curve has, per evaluation, `iteration`, `average` (exploitability of the
    average-policy profile), `nodes_touched` (decisions, the DREAM x-axis: 128 per iteration), `env_steps` (the same
    count), `episodes`, `epsilon`, `seconds`.
  - Reused: `Tree` of `headsup/algos/pdcfr.py`, `TabularPolicy` and `exploitability`.
- `headsup/algos/leduc_report.py`: tables `nfspp` (Leduc by iterations against the paper's Fig. 1a, 64 units),
  `nfspd` (Leduc by nodes against the DREAM paper's NFSP curve) and `nfspk` (Kuhn, no published curve), from
  `runs/leduc_nfsp/{leduc,kuhn}_{paper,dream}_s*.json`.

## Tests (`tests/test_nfsp.py`, `tests/test_deep_algos.py`)

66 tests, under 30 s on one core. 88 one-line mutations of `nfsp.py` were run against them (the list is in the plan):
87 were killed at once; the survivor (`Q` trained before a minibatch was stored: the test's memories were empty) and
two mutants that only an indirect test caught led to stronger tests; all 88 are killed now.

- The formulas: `epsilon` (both schedules), `td_target` (target network values, legal-masked max, nothing added at
  terminals, Double DQN), `cross_entropy` (loss and gradient equal to torch's on masked logits).
- Networks: forward equal to the torch module at every infoset of Kuhn and Leduc; `backward` + `step` equal to
  autograd + `clip_grad_norm_` + `torch.optim.SGD`; one `Q` update and one `Pi` update of the solver equal to the
  textbook version in torch on the same minibatch (both architectures, DQN and Double DQN, with and without clipping,
  a target network that differs from the online one, rows where the best action overall is illegal).
- Memories: FIFO order and wrap-around; the reservoir is uniform over everything offered; `min_prob`; the window.
- Play: deals follow the chance probabilities; the mode is drawn per hand and per seat, independently, with
  probability `eta`, and is held; `M_SL` receives exactly the best-response-mode decisions (exploratory ones too);
  `M_RL` receives every transition of both seats with the next infoset of the same player, the done flag and the
  reward's sign and scale (checked against a step-by-step replay of the game); acting uses each player's own
  networks, the legal-masked argmax, the legal-masked softmax and the schedule's `eps`; the shared exploration coin.
- Each half against exact values: one seat's Q-learning against a fixed opponent reaches the exact best-response
  action values on Kuhn (either seat, both presets); `Pi` learns the eps-greedy behaviour recorded in `M_SL`.
- Schedule: `updates` per network per iteration, `Q` before `Pi`, target refit counted in `Q` updates per player,
  no update before a minibatch is stored, `nodes_touched`.
- Presets set what the table above says; overrides; unknown keywords.
- Checkpoints: a restored solver continues bit-identically; snapshot, source and copy share no arrays; other settings
  are refused; the CLI resumes (the resumed curve equals an uninterrupted run's) and writes atomically.
- Kuhn: both presets reduce the exploitability of the average policy within a few thousand iterations.
- The report tables.

## Success criteria

Three seeds per preset; our value at `x` is the mean of the evaluations within +-10 % of `x`.

| run | published | pass |
|---|---|---|
| Leduc, `paper`, 1e6 iterations | 0.128 | within a factor 1.5 |
| Leduc, `paper`, 3e6 iterations | 0.075 | within a factor 1.5 |
| Leduc, `dream`, 1e8 nodes (781 250 iterations) | 71 mA/g | within a factor 1.5 |
| Leduc, `dream`, 3.2e8 nodes (2 500 000 iterations) | 58 mA/g | within a factor 1.5 |

First curves, measured in pieces of at most ten minutes through checkpoints (mA/g; ours: mean +- sd over seeds):

| Leduc, `paper`, iterations | 1e3 | 1e4 | 1e5 | 2e5 | 5e5 | 1e6 | 1.15e6 |
|---|---|---|---|---|---|---|---|
| paper, Fig. 1a | 1880 | 1040 | 430 | 257 | 158 | 128 | 118 |
| ours, seeds 0-2 | 2136 +- 44 | 1249 +- 82 | 310 +- 50 | 212 +- 78 | 130 +- 54 | 106 +- 39 | 106 +- 39 |

| Leduc, `dream`, nodes | 1.3e6 | 5e6 | 1e7 | 2e7 | 2.6e7 |
|---|---|---|---|---|---|
| DREAM paper, Fig. 2 (NFSP) | 909 | 403 | ~250 | 147 | 127 |
| ours, seed 0 | 841 | 396 | 261 | 167 | 155 |

Kuhn (one seed): `paper` 301 / 109 / 10 / 8 mA/g at 1e3 / 1e4 / 1e5 / 2e5 iterations; `dream` 117 / 128 / 25 at
1e3 / 1e4 / 4e4.

Not verified, and where it may fail:

- `paper` at 3e6 iterations. The runs stand at 1.27e6: 80 / 82 / 158 mA/g, nearly flat since 8e5 (seed 2 since 6e5);
  the criterion is 50-113. The seed-to-seed spread is large.
- `dream` at 1e8 and 3.2e8 nodes. The one run stands at 2.9e7 nodes and has moved between 150 and 164 mA/g since
  2.3e7, where the published curve falls from 136 to 119.

## Compute

One process, one thread (`OMP_NUM_THREADS=1`: more BLAS threads slow these small products down), CPU. Measured on
Leduc while 30-40 threads of other jobs ran on the machine's 32 cores:

| preset | per iteration | env steps / s | SGD updates / s | 3e6 iterations |
|---|---|---|---|---|
| `paper` | 1.3-1.6 ms | 82 000-99 000 | 5 100-6 200 | 1.1-1.3 h (1e6 measured: 0.49 h with the evaluations) |
| `dream` | 8.0-10.8 ms | 11 800-16 000 | 740-1 000 | 6.7-9.0 h (2 500 000, the DREAM curve's end: 5.6-7.5 h) |

`paper` meets the target of three hours; `dream` does not. A `dream` iteration is eight SGD steps on 12-layer
networks at batch 128 plus a second forward pass per `Q` step (Double DQN) and four forward passes for acting: about
3.7 ms of it is single-thread `sgemm` time (64 x 64 products at 13 us each), the rest numpy call overhead on small
arrays. Not done: computing the card and bet branches once per distinct input in a minibatch (10-15 %), and a
GPU port (not measured: this work was CPU-only).

A checkpoint is 20-30 MB (`M_SL`: 2 x 2M rows of 5 bytes); the process uses about 550 MB, most of it the torch import.
