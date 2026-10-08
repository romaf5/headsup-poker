# Deep (Predictive) Discounted CFR on the small games - design

Status: design approved in chat on 2026-10-07; this spec awaits review.
Sources: Xu, Li, Fu, Fu, Xing, Cheng, "Deep (Predictive) Discounted Counterfactual Regret Minimization",
arXiv 2511.08174 (AAAI 2026); the authors' code github.com/rpSebastian/DeepPDCFR @ 9f156c9; the tabular
definitions of Xu et al. 2024 (arXiv 2404.13891).

## Goal

Add the paper's two main algorithms, **VR-DeepDCFR+** and **VR-DeepPDCFR+**, for Kuhn and Leduc, faithful to
the authors' code, and reproduce the paper's exploitability curves on both games.

Non-goals: the hold'em pipeline (FHP; the paper reports only winnings against unpublished rule-based bots), the
paper's other games (Battleship, Goofspiel, Liar's Dice), and its ablation variants beyond the switches below.

## Algorithm

Networks (3 x 64 ReLU MLPs, output layer initialised to zero):

- `R_i` per player: the cumulative *advantage* of each action. It regresses the unclipped quantity; the clip at
  zero is applied wherever the network is read.
- `r_i` per player (VR-DeepPDCFR+ only): the latest iteration's instantaneous advantage, used as the prediction.
- `Q`: one history-action value network of player 0's utility (negated for player 1); input = both players'
  info states. It is only the variance-reduction baseline.
- `Pi`: the average-strategy network (softmax over legal actions). It is the algorithm's output.

Discount: `d_t = (t-1)^alpha / ((t-1)^alpha + c)`; `d_1 = 0`.

One iteration `t` (of `T = episodes / (2K)`), for player `i` in (0, 1) - alternating, so player 1's episodes
already use player 0's updated networks:

1. **Sample `K` outcome-sampling episodes with traverser `i`.** Current strategy at every infoset:
   `sigma_t = RM(R)` (DCFR+) or `RM(max(R, 0) * d_t + r)` (PDCFR+). The traverser samples
   `xi = eps * uniform(legal) + (1 - eps) * sigma_t`, the opponent samples `sigma_t`, chance is sampled.
   Going back up the trajectory, at every decision node `h` (infoset `I`, sampled action `a*`, child `h'`):
   `v(I, a) = Q_i(h, a) + [a = a*] * (v(h') - Q_i(h, a)) / xi(a)`, `v(I) = sum_a sigma_t(a) v(I, a)`;
   terminals give `u_i / scale`.
   - traverser nodes: `(I, v(I, .) - v(I), legal)` goes into this iteration's advantage batch - no reach weight
     and no iteration weight;
   - opponent nodes: `(I, t, sigma_t(I))` goes into the strategy reservoir;
   - every decision node: the transition `(h, a*, u, done, h', I', legal')` goes into the circular baseline
     buffer (`h'` = the history after any sampled deal; `u = u_0 / scale` at terminals, else 0).
2. **Fit `R_i`** on this iteration's batch only: MSE over legal actions between `R(I, a)` and
   `max(R_frozen(I, a), 0) * d_t + advantage(I, a)`, where `R_frozen` is the network as it was before this fit.
   The network and its Adam state persist across iterations. PDCFR+ fits `r_i` on the same minibatches to the
   sampled advantages.
3. **Fit `Q`** from scratch on the circular buffer: target `u + (1 - done) * sum_a' sigma_{t+1}(I', a') *
   Q_target(h', a')`, with `sigma_{t+1}` from the networks just updated (PDCFR+: with `d_{t+1}`) and a target
   network synchronised every 50 steps.

At an evaluation, `Pi` is fitted from scratch on the strategy reservoir with loss weight `(t / T_now)^gamma` and
its exploitability is computed exactly.

Regret matching: proportional to the positive part over legal actions. Fallback when nothing is positive
(`--fallback authors`, the default): DCFR+ plays the legal action with the largest raw network output; PDCFR+
clips the prediction at zero first, so it plays the first legal action. Alternatives: `argmax` (largest unclipped
prediction for both variants) and `uniform`.

## Hyperparameters (defaults = the authors' configs)

| item | value |
|---|---|
| episodes per player per iteration `K` | 10 000 (10 M episodes = 500 iterations) |
| exploration `eps` | 0.6 |
| `alpha`, `gamma`, `c` | DCFR+: 2, 2, 1.5; PDCFR+: 2.3, 2, 1 |
| utility scale | largest absolute utility of the game (Leduc 13, Kuhn 2) |
| optimiser | Adam, lr 1e-3, no gradient clipping |
| hidden-layer initialisation | truncated normal, sd `1 / sqrt(fan_in)`, zero bias |
| `R` / `r` fit | 750 steps x 2 048, minibatches without replacement (the whole batch when smaller) |
| `Q` fit | 1 000 steps x 2 048; circular buffer of 1 000 000 transitions; fitted after each player's update |
| `Pi` fit | 5 000 steps x 2 048; reservoir of 1 000 000; at iterations 1, 2, then every 3rd, and the last |
| loss | MSE averaged over batch x actions, legal actions only |

Where the paper and the code disagree, the default follows the code (it produced the published curves):

| detail | paper | code = default | switch |
|---|---|---|---|
| DCFR+ discount denominator `c` | 1 | 1.5 | `--discount-offset` |
| `Q` steps per fit | 10 000 | 1 000 | `--q-steps` |
| `r` re-initialised every iteration | yes | never | `--reinit-prediction` |
| fallback of the predictive variant | "argmax" | first legal action | `--fallback` |

Deliberate deviations from the authors' code:

1. Their circular buffer under-reports its length after wrapping, which periodically leaves `Q` unfitted. Not
   copied.
2. They keep the `Q` weights of the lowest-loss minibatch; we keep the final weights.
3. Inputs are our info-state encodings (Leduc 34, Kuhn 22 features), not OpenSpiel's tensors.
4. Fits run inside CUDA graphs when a GPU is used (the same arithmetic).
5. `Pi` is also fitted and evaluated at the last iteration (their schedule skips it unless it is a multiple of 3).

## Architecture

- `headsup/algos/pdcfr.py` (new, ~350 lines)
  - `discount(t, alpha, offset)`.
  - `PDCFRSolver(game, variant="dcfr+" | "pdcfr+", ...)` with `iterate()`, `current_policy()`, `policy_net()`,
    `average_policy()`, `evaluate()`, `state_dict()` / `load_state_dict()`. Strategies are tabulated over all
    infosets after each network update and `Q` over all histories after each fit, so an episode is table
    look-ups.
  - Ablation switches that cost nothing: `--no-baseline` (`Q = 0`: the paper's DeepPDCFR+) and
    `--reach-weighted` (the paper's "w/o adv").
  - CLI `python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 10000000 --json ... --checkpoint
    ...`; it prints the same progress lines as `headsup.algos.deep`, plus the episode count.
  - Reused from `headsup/algos/deep.py`: the Adam / CUDA-graph fit helpers, the masked-MSE fit, regret matching,
    `NetPolicy`, the reservoir buffer and exact exploitability. `DeepSolver` itself is not changed.
- `headsup/algos/tabular.py`: tabular `dcfr+` and `pdcfr+` (arXiv 2404.13891) as references:
  `R_t = [R_{t-1} d_t + r_t]^+`; PDCFR+ plays `RM([R_t d_{t+1} + r_t]^+)`; average
  `X_t = X_{t-1} ((t-1)/t)^gamma + reach * sigma_t`.
- `headsup/algos/leduc_report.py`: a table by episodes against the paper's curves (digitised 4-seed means).
- README: the results table and the command.

## Tests (`tests/test_pdcfr.py`, `tests/test_games_algos.py`)

- The discount schedule (`d_1 = 0`, both offsets) and the bootstrapped target (clip at read time, not in the
  target).
- On Kuhn, the mean sampled advantage at each traverser infoset equals the exact advantage of the current
  strategy (by enumeration), with `Q = 0` and with a random `Q` (unbiasedness with any baseline).
- The predictive strategy rule and each fallback mode.
- The baseline target on a hand-built transition (terminal, non-terminal, opponent to act).
- Kuhn smoke run: exploitability of `Pi` falls below a threshold within a small budget; checkpoint resume
  continues the iteration and episode counts.
- Tabular DCFR+ / PDCFR+ converge on Kuhn (game value -1/18) and on Leduc (exploitability below a fixed bound
  after a fixed number of iterations).

## Success criteria

Unit: mA/g = 1000 x NashConv / 2 with ante 1 (the paper's unit x 1000; its Kuhn and Leduc equal ours). Three
seeds per variant; "final" = the mean of the evaluations between 9 M and 10 M episodes.

| game, variant | paper (4-seed mean) | pass |
|---|---|---|
| Leduc, VR-DeepDCFR+ | 89 (95 % band 51-133) | 3-seed mean inside the band |
| Leduc, VR-DeepPDCFR+ | 90 (95 % band 63-147) | 3-seed mean inside the band |
| Kuhn, VR-DeepDCFR+ / VR-DeepPDCFR+ | 5.3 / 3.3 | within a factor of 2 |

The curves are also compared at 1 M / 2 M / 4 M / 8 M episodes (paper, Leduc: DCFR+ 152 / 121 / 114 / 85,
PDCFR+ 158 / 121 / 115 / 88). The paper's own baselines at 10 M episodes: DREAM 153, OS-DeepCFR 313.

## Compute

Kuhn runs take minutes. A Leduc run is about 2-3 hours (10 M Python episodes plus roughly 3 M SGD steps); the
six Leduc runs go in parallel on one GPU and ~12 cores, 3-4 hours of wall time (approved 2026-10-07).
