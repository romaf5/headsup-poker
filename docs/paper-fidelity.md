# Paper fidelity

What this code does differently from the papers it implements, and what was changed to match them.
Source: an audit on 2026-10-07/08 of every algorithm against its paper and, where it exists, the
authors' code (Deep-CFR / PokerRL, DREAM, ESCHER, OpenSpiel, DeepPDCFR). Each item is one of

- **fixed** - the code now does what the paper (or the authors' code) does; the test that pins it is named;
- **by choice** - paper and authors' code disagree, or we deviate on purpose; what we follow and why;
- **open** - a known difference that is still there, with what is known about its effect.

No bug was found in the core estimators: the external-sampling, DREAM and ESCHER samplers (small games,
hold'em Python, hold'em C++), the tabular solvers, the best-response evaluators and the hand evaluator were all
checked against independent implementations or enumeration.

## Games

| item | status |
|---|---|
| FHP / HULH betting: four bets per round. Pre-flop the big blind is the first, so 3 raises; post-flop a bet and 3 raises. Our flop had a bet and 2 raises (Appendix A's "at most three raises" read literally). Only the four-bet flop gives the abstraction sizes in Deep CFR's Fig. 2 (169 x 21 + buckets x 182 = 39,949 / 367,549 / 3,643,549 infoset-actions; 234,199,693 lossless) and HULH's 3.19e14 infosets (Bowling et al. 2015). | **fixed** - `FHP` caps (3, 4), `HULH` (3, 4, 4, 4); `test_fhp_is_the_game_of_the_deep_cfr_paper` |
| Exact tabular Linear CFR on FHP against the paper's dashed reference line (total exploitability, mbb/g): 115 / 64.7 / 31.9 at iterations 20 / 30 / 50, paper 118 / 65 / 31. At iterations 5 / 10 ours is 4 / 11 % lower (976 / 355 vs 1018 / 398); convention unknown. | checked (`python -m headsup.algos.fhp_cfr`) |
| A betting round can have at most 6 actions - the observation's history slots, as in the paper ("at most 6 sequential actions"). | **fixed** - `GameConfig` rejects games with longer rounds |
| Kuhn / Leduc: isomorphic to OpenSpiel history by history; exploitability equal to 1e-15. | checked |

## Deep CFR and Single Deep CFR (hold'em trainer, `headsup.deepcfr.train`)

| item | status |
|---|---|
| Network. The paper's "98,948 parameters" is the family 23 d² + 74 d + 4 (it also matches the model sizes of Fig. 3): Appendix C with the card branch 3 d wide, as in both of the authors' code bases. `--net paper` had a d-wide card branch (67,459 parameters), 51 bet inputs instead of the bet history alone, and a learned "no card" row. | **fixed** - `--net deepcfr` (`--preset paper` selects it); `test_deepcfr_paper_network`. It keeps Appendix C's 52-card tables, which the count leaves out (rank + suit alone cannot tell A♠K♥ from A♥K♠), and FHP's 3 outputs instead of 4 |
| Loss weights: "we rescale all the batch weights by 2/T" (5.3). We used the raw t, so the gradient clip at 1 rescaled every step (norms 25-460 at T = 450). | **fixed** - `--loss-weights paper` (default); `raw` for the old behaviour, which resumed older checkpoints keep |
| Refits start "from a random initialization" (5.2); only the first network must output zeros. Every refit started with a zero head. | **fixed** - `test_refits_start_from_a_random_head` |
| First strategy: with the untrained network and the argmax fallback we played the first legal action (fold facing a bet); the authors play uniformly before the first network exists. | **fixed** - exact ties share the probability, in C++, numpy, torch and the small-game solver; `test_untrained_networks_play_uniformly_in_the_first_iteration` |
| Average-strategy network: StepLR x 0.9 every 2 % of the fit (20 % of a constant schedule's learning-rate integral), no gradient clipping, softmax over illegal actions too. Authors: constant rate, clipping, illegal logits masked. | **fixed** - constant rate and clip by default, masked softmax with `--masked-loss`; `test_policy_fit_follows_the_authors` |
| SD-CFR average: "each D^t is assigned sampling weight t" (5.1, authors' code). Our bank gave the untrained network weight 1 and the network of iteration t weight t + 1, and `@tN` stopped one iteration short. (The paper's eq. 4-5 imply t + 1; its text and code use t.) | **fixed** - `test_bank_weights_follow_the_authors` |
| Illegal actions' outputs: the paper is silent, the authors' networks multiply them by the legal mask. | **by choice** - `--masked-loss` is off for Deep CFR / SD-CFR (no difference measured on FHP), on for DREAM / ESCHER |
| One deal per traversal shared by all of the traverser's branches (the authors reshuffle per branch). | **by choice** - both unbiased, ours has lower variance |
| Strategy memory: one memory and one policy network for both players (the paper's Algorithm 1); SD-CFR's code has one per player. | **by choice** - the paper |
| Regret scale: the Deep-CFR repository divides external-sampling regrets by the number of legal actions; the paper and the DREAM repository do not. | **by choice** - the paper (the small-game solver has `--mean-regret`) |
| Reservoir: Algorithm R (the authors' acceptance test is off by one). | **by choice** |
| Regret targets are fitted in units of their RMS and the output layer is scaled back (`--target-scale auto`). Not in the paper; without it Adam cannot reach FHP's regrets of hundreds of chips from a small head. | **by choice** |

Open, about the paper's own figures:

- Its FHP figures are not mutually consistent with "K = 10,000 traversals, 4,000 SGD steps": Fig. 2 is the K = 10k line of
  Fig. 3-left (154 / 70 / 51 / 40 / 40 mbb/g at t = 50 / 100 / 200 / 300 / 450), while the SGD-step sweep and Fig. 4 start
  much lower (231-415 at t = 10 against 770), i.e. were run at 1e5-1e6 traversals. All K from 3k to 1M end at 34-46.
- It touches 0.94-1.23e6 nodes per iteration at K = 10k; we measured 2-5e5 on the old three-bet flop. To be re-measured on
  the four-bet game.
- **The FHP reproduction itself is open.** The earlier 80 mbb/g (total, 450 iterations) was measured on the three-bet
  flop with the narrow network and raw weights; it has to be re-run.

## DREAM and ESCHER

| item | status |
|---|---|
| Hold'em defaults: uniform fallback and illegal outputs fitted to the samplers' zeros. DREAM 5: "picking the action with the highest advantage with probability 1 when all are negative"; both authors' networks mask. | **fixed** - argmax and masked losses are the defaults for `--algo dream / escher` |
| ESCHER budgets: Table 3 has 1,000 regret and value trajectories, batch 2,048, 5,000 regret / 5,000 value / 10,000 policy steps; our defaults give the value net 20x fewer sample draws. | **fixed** - `--preset escher` in both trainers |
| ESCHER's value net is refitted on its iteration's trajectories; ours passed them through the DREAM FIFO (`--q-capacity`), which kept the newest rows only. | **fixed** - `test_escher_value_net_sees_all_trajectories_of_its_iteration` |
| ESCHER's tabular Leduc experiment (Fig. 3): we reached NashConv 0.10 (ESCHER) and 0.15 (DREAM) where the paper has ≈0.54 and ≈0.37. The difference is the average-strategy accumulator: OpenSpiel's outcome sampling adds own reach x sigma / sampling reach at the update player's sampled infosets. | **fixed** - `--average own_is` gives 0.87 / 0.28 / 0.51 (ESCHER) and 0.51 / 0.48 / 0.48 (DREAM) over 3 seeds; the exact average stays the default. The variance pooled over an iteration's estimates (the paper's definition) is 5.1 and 312 against 5.3 and 280 |
| First strategy uniform (DREAM 4). | **fixed** - see above |
| DREAM sample weight: 1 / sampling reach (paper 5.1) against 1 / (sampling reach x number of legal actions per step) in the code. | **by choice** - the paper; in Leduc both reach the same fit accuracy (relative RMS error 0.148 / 0.144) |
| DREAM baseline: one per player (paper) or one shared network of player 0's values trained once per iteration (code); bootstrapping across deals (code). | **by choice** - the paper by default, the code with `--shared-baseline --bootstrap-chance` (what reproduces the Leduc curve) |
| ESCHER: q(h, a) network and r = q - sum pi q (paper) against a scalar V(h) evaluated at every child through the simulator (code); no importance sampling (paper) against the code's default; average-policy rows = the full sigma at opponent nodes (code). | **by choice** - paper, paper, code |
| DREAM baseline targets are computed when the transition is sampled (reward + sigma(h') Q_old(h')); the paper's expected SARSA and the authors' code recompute the bootstrap from the network being trained in every minibatch. | **open** - Leduc smoke test: baseline RMSE 4.2 / 3.5 / 3.1 against 3.1 / 3.4 / 2.5 in iterations 1-3, the same afterwards; not measured on deep trees. Deep PDCFR+ (`headsup.algos.pdcfr`) implements the bootstrapped version |
| `--arch deepcfr_dueling` body: 64-wide card branch without skip, no skip in the bet branch, three trunk layers with the skip outside the ReLU, normalisation without gain / bias. Authors: 192-wide card branch with skip, history skip, two trunk layers relu(Wz + z), LayerNorm with gain and bias. | **open** - low; the Leduc curves are reproduced with it |
| ESCHER networks: 3 x 64 ReLU with gradient clipping 1; the reference has (256, 128) LeakyReLU + LayerNorm and no clipping. | **open** |
| Small-game reward unit: antes; the authors divide by 2.6. Baseline loss: mean over batch x actions (1/3 of the authors'); 200k rows drawn with replacement before the fit. `--loss-weights normalized` divides by t (the authors by the last stored weight). | **open** - low |
| Missing variants: DREAM with an average network, periodic / never reset. | **open** |

## Small-game deep solvers (`headsup.algos.deep`)

| item | status |
|---|---|
| The "current" exploitability column used the uniform fallback while the solver plays argmax. | **fixed** - `test_current_policy_is_the_strategy_the_solver_plays` |
| ESCHER's nodes-touched count omitted terminal states. | **fixed** |
| `--arch deepcfr`: skip added after the ReLU, bet branch without skip (Appendix C: relu(W z + z)). | **open** - low |
| The SD-CFR paper may count alternating single-player updates as iterations (its Leduc x-axis would then be twice ours). | **open** - not resolvable from the sources; our curves match the paper's with one iteration = both players |

## Tabular CFR family (`headsup.algos.tabular`)

| item | status |
|---|---|
| Vanilla CFR, CFR+, Linear CFR, DCFR: equal to OpenSpiel's solvers to 1e-11. MCCFR external / outcome sampling: unbiased against enumeration. | checked |
| DCFR+ / PDCFR+ (Xu et al. 2024): PDCFR+ predicted with the discount of the current iteration for a player already updated in it. | **fixed** - equal to an independent implementation of the paper's equations to 2.5e-13 |
| PCFR+: the prediction is the last instantaneous regret (as Xu et al.), not Farina et al.'s construction (Leduc at 1000 iterations: 7.1e-4 against 3.3e-4). | **open** |
| Small-game MCCFR pruning: the 95 % draw is per node, without Pluribus's exemptions for the last round and terminal-leading actions, and without a regret floor. | **open** |

## Deep (Predictive) Discounted CFR (`headsup.algos.pdcfr`)

Written against the authors' code; the deviations from the paper's text and from the code are listed in
`docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md`. An independent review compared the sampler, the strategy
rule, both fits and the average-policy fit with the authors' code (120k Leduc episodes, 20,000 random strategy cases,
losses equal to 15 digits) and found no difference.

## Pluribus-style blueprint, search and LBR

Consistent with the paper: the MCCFR-P update rule, regret matching, the shape of the pruning rule and the floor ratio,
the linear discount, external sampling, blueprint play (average strategy), the search root at the start of the round,
the belief updates, final-iterate play in Pluribus mode, the vector Linear CFR, DCFR's constants, and LBR's action
values (equal to the paper's Fig. 1 formula to 5.5e-14).

Open (the next batch of work; these change how the `pluribus` bot plays and need a new blueprint and new measurements):

1. **Bucket assignment is stochastic.** The flop / turn bucket of a (hand, board) comes from 500 Monte-Carlo runouts per
   call: over 40 seeds it has a standard deviation of 5 of 200 buckets and lands in its modal bucket 16 % of the time.
   The paper puts every situation into one bucket. Fix: a deterministic mapping (cached per board, or a seeded estimate).
2. **Search re-solves at every decision of a round** and freezes its own earlier action as a one-hot row, which then
   inflates the real hand's weight in its own range by 1 / sigma(a). The paper solves once per round and freezes
   sigma(I). 8.5 % of post-flop rounds are affected.
3. **The opponent's belief starts conditioned on the searcher's hole cards**; the paper keeps public beliefs (1/1326 each).
4. **Later-round bucket regrets in the vector solver ignore the solving player's own range weights.**
5. **LBR against a search player queries its blueprint**, not the searched strategy (total-variation distance 0.40 between
   the two). The bound stays valid but is not the paper's LBR; needs 2 and 3 first.
6. **Pruning never fires at our budgets**: the threshold (-3e6) is scaled by the stack only; after 20 M iterations the most
   negative regret is -1.39e6. The shipped blueprint is Linear MCCFR without pruning.
7. **Average strategy**: reach-weighted at every sampled opponent infoset on every round (better in our measurement:
   508 / 541 against 629 / 663 mbb/g with Pluribus's counters every 10,000 iterations), not Algorithm 1's round-1
   counters plus later-round snapshots; `--strategy-every` has no effect under this default.
8. **Depth-limited mode** (`search:<net>` without `@pluribus`): one sampled continuation by default, where both papers
   warn about fixed leaf values and use four; it plays the average rather than the final iterate; `@leaf4` cannot be used
   with a tabular blueprint.
9. Linear-CFR / pruning schedule: 40 % / 20 % of the run against Pluribus's 3.5 % / 1.7 %, and the help text misstates
   Pluribus's.
10. LBR: flop equities from 200 Monte-Carlo runouts (paper: exhaustive), no round-restricted variants.
