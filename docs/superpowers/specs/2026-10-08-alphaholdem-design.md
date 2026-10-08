# AlphaHoldem on the default no-limit game - design

Status: written and implemented by one agent under a blanket approval (2026-10-08); the long GPU run and the review
are still to come.
Sources: Zhao, Yan, Li, Li, Xing, "AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker
via End-to-End Reinforcement Learning" (AAAI-22); the same lab's OpenHoldem (arXiv 2012.06168v4), which re-describes
the agent; Ye et al. 2020 (arXiv 1912.09729) for the dual-clip PPO loss. There is no official code; the public
re-implementations (bupticybee/AlphaNLHoldem, nganteehee/AlphaHoldem, EircJC/AlphaHoldem-HU) were read for the
choices they made, not copied.

## Goal

The paper's three components - the card / action tensors with a pseudo-siamese network, Trinal-Clip PPO and K-Best
self-play - for this repository's default game (`headsup.game.DEFAULT_GAME`: stacks 100, blinds 1/2, fold /
check-call / min-raise / all-in, the third raise in a row is an all-in), where the shipped baselines are (`cfr`,
`tab`, `pluribus`). A larger action tree only changes tensor sizes.

Non-goals: the paper's 9-action 200-bb game and its Slumbot / human results, a distributed (MPI, multi-GPU) trainer,
a replay buffer, the Leduc sanity run the brief suggests (a possible follow-up), any search at play time.

## Pieces

### Two-seat vectorised environment (`headsup_cpp.SelfPlayVecEnv`, `headsup/twoseat.py`)

The existing envs seat one agent against an opponent that acts inside the env; K frozen conv-nets cannot. The new
env drives both seats from Python in lock-step:

- `reset(decks=None) -> (obs[N, 80], seat[N])`: every table is dealt; the observation is the one of the seat to act
  (always seat 0 after a deal).
- `step(actions, decks=None) -> (obs, seat, rewards[N, 2], dones[N])`: one action per table for the seat to act. When
  a hand ends, `rewards[i]` holds both seats' chips, `dones[i]` is set, the table is dealt again and `obs[i]` is the
  first observation of the new hand. `actions[i] < 0`: the table waits (nothing changes; the same observation comes
  back) - the trainer uses it to let the hands in progress finish at the end of a rollout.
- `decks` (int[N, 9]): the cards of a table's next deal, for tests; otherwise the env's own generator deals.
- `allin_ev` / `ev_samples`: rewards of hands that end all-in before the river are the expectation over the runouts
  (as in `VecEnv`); the sampling seed of a hand is a hash of (env seed, table, hand number), the same in both
  implementations.
- No opponent acts during a reset, so the one-seat envs' "terminal observation after an open-fold" cannot occur here.

`headsup/twoseat.py` holds the Python twin (`TwoSeatVecEnv`, on `HeadsUpPoker` engines, seeded like `PokerVecEnv`
so the same seed deals the same cards), the wrapper of the C++ class and `make_two_seat_env`.

### Tensors from the observation (`headsup/alphaholdem/encoding.py`)

The player must be a function of the float32[80] observation alone (`headsup.lbr` queries strategies on
hand-substituted observations), so both tensors are rebuilt from it, batched, on the training device.

- **Card tensor** bool[6, 4, 13] (suit x rank): hole cards, flop, turn, river, all public cards, hole + public.
- **Action tensor** bool[24, 4, A] with `A = game.num_actions` (4 here; the paper's nb = 9): channel
  `6 * round + slot` is the slot-th decision of a betting round; rows: seat 0's action (one-hot over the A actions),
  seat 1's action, their sum, and the legal actions *at that decision* (the paper's Fig. 3). The decision the player
  now faces has only its legal row set, in the next free slot of the current round.
- **Replay.** obs[31:79] stores, per round and slot, `chips put in / pot before` and `occurred`. Heads-up actions
  alternate from the round's first actor (seat 0 pre-flop, seat 1 after), so actor, pot, bets, stacks, the call
  amount and the raise counter before every slot follow from cumulative products / sums over the slots - no loop.
  The executed action: check/call when the chips equal the call amount, fold when they are 0 facing a bet (terminal
  observations only), all-in when they equal the stack, else the first raise size whose amount matches. Raise
  amounts come from a table built with `GameConfig.raise_amount` (indexed by pot + call), so the rounding is the
  engine's own; legal rows are `GameConfig.legal_mask` vectorised.
- **Legal mask of the current decision** from obs[21, 23, 28, 29, 79], equal to `legal_mask_from_obs`. It masks the
  policy logits and fills the pending slot's legal row.
- **Value-clip bounds** from obs[24, 25, 29]: the chips the observer and the opponent have put in so far.

Limits: a round's 7th action is not recorded (the cap of the default game allows 5); in a tree without
`mask_redundant` a raise that the cap or the stack turned into an all-in appears as the all-in (it is one).

### Network (`headsup/alphaholdem/model.py`)

Two convolutional towers without shared parameters, one per tensor: `conv_layers` (3) x [3x3 conv, `channels` (64),
ReLU], flatten, linear to `hidden` (256), ReLU. Their outputs are concatenated (512), one more linear + ReLU (256),
then a policy head (A logits; illegal actions are set to -1e9 before the softmax) and a value head. About 1.4 M
parameters. Orthogonal initialisation (gain sqrt 2; policy head 0.01, value head 1). No normalisation layers.
Files are `{"config", "state_dict"}` with `config["game"]` = the action tree, like every other model here.

### Trinal-Clip PPO (`headsup/alphaholdem/ppo.py`)

With `r = pi(a|s) / pi_old(a|s)` and advantage `A`:

```
surr   = min(r * A, clamp(r, 1 - eps, 1 + eps) * A)          # PPO
surr   = where(A < 0, max(surr, delta1 * A), surr)           # the third clip: bounded when A < 0 and r is large
target = clamp(R, -delta2(s), delta3(s))                     # R: the discounted return of the hand
loss   = -mean(surr) + c_v * mean((target - V(s))^2) - c_e * mean(entropy)
```

For `A < 0` this is `clamp(r, 1 - eps, delta1) * A`, which is what the paper's eq. (3) evaluates to; for `A >= 0` it
is PPO. `delta2(s)` / `delta3(s)` are the chips the player / the opponent have put in *up to the state s*
(per-state reading), divided by the reward scale.

### Rollouts and advantages

One iteration:

1. Every table gets an opponent for the iteration: the current agent or one of the pool's survivors, each with the
   same number of tables (a random assignment). Against a pool member the main agent's seat alternates from hand to
   hand; at a table of the current agent against itself both seats are the main agent.
2. Lock-step: the main network acts for its seats (actions sampled from the masked softmax; log-probability and value
   are stored), each pool member acts for its tables (frozen, no gradient). This continues until `--samples` main
   decisions are collected; then the tables whose hand is in progress finish it while the others wait.
3. So a rollout holds complete hands only. Advantages are GAE(gamma, lambda) along each (table, seat) stream of main
   decisions; the hand's reward (chips / `--reward-scale`) arrives at the stream's last decision of the hand and
   nothing is bootstrapped past the hand's end. The value target is the discounted return of the hand. A hand in
   which a seat never acted (the other seat open-folded) contributes no sample for that seat; its reward still counts
   in the results against the pool.
4. `--epochs` passes over the samples in minibatches of `--minibatch`, Adam.

### K-Best self-play (`headsup/alphaholdem/pool.py`)

- Every `--snapshot-every` iterations a frozen copy of the main agent joins the pool with the main agent's current
  ELO. The pool keeps the `--pool` (K) members with the highest ELO.
- ELO: per iteration and pool member, the hands the main agent played against it are one game - won (1), drawn (0.5)
  or lost (0) by the sign of the chip total; standard update with K-factor 16 for both sides, scale 400.
- Opponents are sampled uniformly from the survivors plus the current agent.

### Trainer, player, tools

- `python -m headsup.alphaholdem.train --out runs/x --iterations N --device cuda:0 [--resume]`: writes
  `checkpoint.pt` (network, optimiser, pool, ELO, generators, log), `policy.pth` (the player's file) and `log.json`;
  evaluates every `--eval-every` iterations against `random` / `call` / `allin` (and `cfr` with `--eval-cfr`) with
  all-in EV rewards, mean ± SE.
- `headsup.alphaholdem.player.AlphaHoldemPlayer`: stateless, `probs(obs, ids=None)`, `game`; samples (argmax with
  `deterministic`). Spec `alpha:<path>` in `make_player`, so `headsup.compare`, `headsup.lbr` and the browser table
  take a trained agent.

## What the paper leaves open, and the choice made

| # | point | paper | choice | why |
|---|---|---|---|---|
| 1 | tower / FC layers | "ConvNets", "like ResNet", 8.6 M parameters (1.8 M conv, 6.8 M FC) | 3 plain 3x3 conv layers x 64 per tower, 256-unit layers, 1.4 M parameters | the game is 4 actions x 50 bb instead of 9 x 200 bb; the brief's sizing; all widths are flags |
| 2 | eps | "typical value 0.2" | 0.2 | |
| 3 | form of eq. (3) | no min, no sign condition; text: delta1 applies "when A < 0" | dual-clip of Ye et al. (above) | it is what eq. (3) gives for A < 0, and the ablation ladder (PPO < dual-clip < trinal) puts the value clip on top of dual-clip |
| 4 | delta2 / delta3 | "the total number of chips the player has placed and the opponent has placed", "dynamically calculated" | per state: chips put in up to s | OpenHoldem: "-delta2 represent the state value when the player folds, delta3 ... when the opponent folds"; the per-hand reading (final chips) never clips a complete hand's return |
| 5 | which return is clipped | "the traditional gamma-return" | the discounted Monte-Carlo return of the hand | literal; rollouts hold complete hands, so it is available |
| 6 | total loss, coefficients | not given | value 0.5, entropy 0.01 | PPO defaults; the brief's values |
| 7 | epochs, minibatch | minibatch 16 384, epochs not given | 3 epochs, 16 384 | |
| 8 | advantage normalisation | not given | mean / std over the rollout (`--no-adv-norm` turns it off) | PPO practice; without it the policy term scales with the reward scale |
| 9 | gradient clipping, LR schedule | "initial learning rate 0.0003" | global norm 0.5, constant 3e-4 | PPO practice; no schedule is described |
| 10 | reward | "the game reward" | chips / 100 (the stack); `--allin-ev` optionally replaces all-in runouts by their expectation | targets in [-1, 1]; the paper names no variance reduction, so the default is the dealt outcome |
| 11 | gamma, lambda | 0.999, 0.95 | same, per decision of the acting player | |
| 12 | rollout length | 128 steps x 128 envs x 8 workers | complete hands, at least 131 072 decisions per iteration | no truncated hands and no bootstrap value at a point where the opponent is to act |
| 13 | K, snapshot interval | not given | K = 8, every 50 iterations | the brief's values; flags |
| 14 | opponent sampling | Fig. 4(f): arcs among the pool and the current agent | uniform over survivors + current agent, per table and iteration | |
| 15 | ELO from poker results | not given | one game per (pool member, iteration) by the sign of the chip total, K-factor 16 | a per-hand win / loss ignores pot sizes |
| 16 | pool-vs-pool games | not given | none; pool members are rated by their games against the main agent only | "trains only one agent" |
| 17 | legal mask of the current decision | Fig. 3 shows the legal row of a past decision | the pending slot's legal row; logits masked | |
| 18 | hero's seat | rows are player 1 / player 2 | absolute rows (seat 0 = small blind); the seat follows from the pending slot | Fig. 3 |
| 19 | check vs call columns | separate | one check/call column | this repository's action set |
| 20 | pot / stack inputs | none (fixed stacks) | none | |
| 21 | test-time action | not given | sampled (`deterministic` = argmax) | a mixed strategy is the point in poker |
| 22 | which checkpoint | "smaller overall loss generally performs better" | the last main agent | |

## Deviations from the paper

1. The game: 4 actions, 50 bb, the repository's raise cap, instead of 9 actions and 200 bb.
2. One process, one GPU, synchronous on-policy PPO; the paper's "off-policy ... replay buffer" with 8 GPUs is not
   described well enough to copy.
3. Complete-hand rollouts and Monte-Carlo value targets (rows 5 and 12).
4. Tensors are rebuilt from the observation instead of being written by the engine (the same content; tested against
   the engine's history).
5. The network is about one sixth of the paper's size.
6. Optional all-in EV rewards (off by default).

## Tests (`tests/test_twoseat.py`, `tests/test_alphaholdem.py`; CPU, under a minute together)

- Two-seat env: C++ against the Python twin on the same decks with random actions including waits (observations,
  seats, rewards, dones identical; also with all-in EV); the twin against `PokerVecEnv` with the same seed and
  deterministic players (the same rewards hand by hand); waiting tables; out-of-range actions.
- Tensors: against an independent construction from the engine's true history (actor, executed action, legal set at
  each decision) on random hands in three trees; the card tensor; the legal mask against `legal_mask_from_obs`; the
  bounds against the engine's chips; terminal observations do not break the encoder.
- Loss: the three clips on hand-computed cases, gradients included; the delta1 clip differs from PPO where it should
  (and a `min` of three terms does not); the value target with per-state bounds.
- GAE on a hand-built two-table trajectory: interleaved seats, a hand without a decision of the learner, waiting
  cells; against values computed by hand.
- ELO update, K-best selection, opponent assignment.
- Trainer: frozen pool members do not change, the main agent does; samples come from main-controlled seats only and
  every sample's return is its seat's reward of that hand; checkpoint round trip; a short run logs finite losses.
- Player: probabilities sum to 1, no illegal action, batch = row by row, `make_player("alpha:...")`, LBR and compare
  accept it.
- Mutation check (by hand, recorded in the plan): each wiring test fails on its one-line mutation.

## Compute

- CPU smoke (8 threads): a few minutes; must beat `random` and not lose to `call`.
- Long run (one RTX 3090): 5 000 iterations x 131 072 decisions = 0.66e9 samples, the paper's ablation budget.

## Success criteria (long run; 400k hands head-to-head, >= 20k LBR pairs, ± SE)

1. Positive against `random`, `call`, `allin`.
2. Not losing to `cfr` (`models/deepcfr_policy.pth`); stretch: the blueprint's +0.87 ± 0.04 chips/hand against it.
3. LBR below the DeepCFR net's 1.33 ± 0.09 chips/hand; stretch: the blueprint's 0.60 ± 0.09.

The paper reports no exploitability, and self-play PPO has no convergence guarantee in imperfect-information games
(AlphaExploitem reports that a K-best PPO baseline approaches but does not reach Nash on Leduc), so criterion 3 is
the open one.
