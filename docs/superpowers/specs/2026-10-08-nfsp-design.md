# Neural Fictitious Self-Play on the small games - design

Status: written and implemented autonomously on 2026-10-08 (the work was pre-approved); the long validation runs are
still to be done.
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
| reward unit | antes (`scale` 1) | antes / 2.6 (`scale` = largest utility / 5; Kuhn: 0.4) |
| exploration | `0.06 / sqrt(t)`, one coin per decision | `0.06 / (1 + 0.01 sqrt(t - 1))`, one coin per seat and step for all tables |
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
| reward scale | not stated (antes assumed) | 1 | stack / 5 | 1 | largest utility / 5 | `--reward-scale` |
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

## Deliberate deviations

From all three sources:

1. **Inputs.** The repository's perfect-recall info-state features (Leduc 34, Kuhn 22), not the paper's 30-feature
   card / betting tensor, OpenSpiel's tensor or PokerRL's observation.
2. **The networks are numpy arrays with a hand-written backward pass.** An SGD step on a 64-unit network is bounded
   by per-operation overhead, not arithmetic: on this machine a `Q` step costs 480-570 us with torch autograd against
   about 75 us in numpy for the 1 x 64 MLP, and 3.5 ms against about 0.6 ms for the dueling net. The parameters keep
   the torch module's names and layout, the initial weights are that module's, and the tests require the forward pass
   and one whole `Q` / `Pi` update to equal the torch module's with autograd and `torch.optim.SGD`. `--device` exists
   for symmetry with the other solvers and accepts `cpu` only (a GPU was not measured: it is launch-bound too).
3. **The target network is stored as its outputs at every infoset** (one forward pass over the game's infosets per
   refit instead of one per minibatch) - the same numbers.
4. **Memories hold infoset indices**, not observations (the tree is compiled to arrays): `M_SL` is 5 bytes a row.
5. **The reservoir is the textbook Algorithm R** (item `m` is kept with probability `capacity / m`); the DREAM code
   divides by the number of items seen before it.
6. **A zero standard deviation in the dueling net's normalisation gives a zero gradient**, where torch gives NaN.

`paper` preset, from the paper:

7. 128 parallel tables with one step each per iteration instead of (presumably) one game played sequentially: a hand
   spans several iterations, the data are the same.
8. `Q` and `Pi` are not updated before their memory holds a minibatch (the paper does not say).

`dream` preset, from the DREAM code:

9. **Network body.** `deepcfr_dueling` is the repository's net (docs/paper-fidelity.md): a 64-wide card branch without
   skip connection, three trunk layers with the skip added after the ReLU, normalisation without gain and bias; the
   authors' has a 192-wide card branch and a history branch with skips, two trunk layers and a LayerNorm with gain and
   bias. Its `Pi` head has a hidden layer (64) before the logits; the authors' is a single linear layer.
10. **Kuhn's reward scale** is the largest utility / 5 = 0.4 by analogy (PokerRL has no Kuhn).
11. Iteration 0 of the DREAM driver skips the `Pi` updates; here they start when `M_SL` holds a minibatch (iteration
    ~20), which subsumes it.

Not copied from OpenSpiel: the inner DQN's counters that only advance in best-response-mode steps, the soft
`M_SL` targets, the minimum buffer size of 1000.

## Architecture

- `headsup/algos/nfsp.py` (new)
  - `epsilon(t, start, const)`, `td_target(reward, done, next_target, next_legal, next_online=None)`,
    `cross_entropy(logits, legal, action)` - the three formulas, as pure functions.
  - `MLP` / `Dueling`: the numpy networks (`forward`, `backward`, `step`, `state_dict`, `torch_module`).
  - `Memory`: named columns, circular or reservoir (with `min_prob`).
  - `NFSPSolver(game, preset="paper", seed=0, **overrides)`: `iterate()`, `average_policy()`, `evaluate()`,
    `state_dict()` / `load_state_dict()` (the whole state: networks, target values, memories, tables, modes, pending
    transitions, counters, random generator; a snapshot shares nothing with the solver; a checkpoint written with
    other settings is refused).
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

Each wiring test was checked against the one-line mutation it is meant to catch.

- The formulas: `epsilon` (both schedules), `td_target` (target network values, legal-masked max, nothing added at
  terminals, Double DQN), `cross_entropy` (loss and gradient equal to torch's on masked logits).
- Networks: forward equal to the torch module at every infoset of Kuhn and Leduc; one `Q` update and one `Pi` update
  of the solver equal to torch autograd + `torch.optim.SGD` on the same minibatch (both architectures, DQN and Double
  DQN, with and without clipping, a target network that differs from the online one).
- Memories: FIFO order and wrap-around; the reservoir is uniform over everything offered; `min_prob`; the window.
- Play: the mode is drawn per hand and per seat, independently, with probability `eta`, and is held; `M_SL` receives
  exactly the best-response-mode decisions (exploratory ones too); `M_RL` receives every transition of both seats with
  the next infoset of the same player, the done flag and the reward's sign and scale (checked against a step-by-step
  replay of the game); acting uses each player's own networks, the legal-masked argmax, the legal-masked softmax and
  the schedule's `eps`; the shared exploration coin.
- Schedule: `updates` per network per iteration, `Q` before `Pi`, target refit counted in `Q` updates per player,
  no update before a minibatch is stored, `nodes_touched`.
- Presets set what the table above says; overrides; unknown keywords.
- Checkpoints: a restored solver continues bit-identically; snapshot, source and copy share no arrays; other settings
  are refused; the CLI resumes and writes atomically.
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

## Compute

One process, one thread, CPU. See "Measured throughput" in the plan for the numbers of the finished implementation.
