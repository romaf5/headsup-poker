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
- It touches 0.94-1.23e6 nodes per iteration at K = 10k (Fig. 2 against Fig. 3-left, from the first iteration on). On the
  four-bet game an external-sampling traversal under uniform play visits 24.6 histories as the small blind and 16.0 as
  the big blind (decision, terminal and flop chance nodes; 5.0 / 3.4 of them are the traverser's infosets), i.e. 4.1e5 per
  iteration at K = 10k per player, and 4.6-5.5e5 during training: the paper's "10,000 traversals" touch 2.2-2.5 times
  more nodes than ours. The tree is the same (its infoset counts are reproduced), so either the count or K means
  something else there.
- Its SGD-step sweep (Fig. 3-middle) ends at about 65 mbb/g for 4,000 steps (110 / 80 / 65 / 43 / 37 / 35 for 1k ... 32k),
  the traversal sweep (Fig. 3-left, nominally 4,000 steps) at 34-46 for every K: the headline 37-40 is the level of the
  sweep's 8,000-32,000-step runs.
- **The FHP reproduction itself is open.** Runs of 2026-10-08 on the four-bet game with `--preset paper` (total
  exploitability in mbb/g; policy net by exact best response over all flops, SD-CFR average on 2,000 sampled flops,
  which overstates by ~8 %):

  | | t = 50 | 100 | 200 | 300 | 450 |
  |---|---|---|---|---|---|
  | paper, Fig. 3-left, K = 10,000 | 154 | 70 | 51 | 40 | 40 |
  | paper, Fig. 3-middle, 4,000 SGD steps (many more traversals) | 75 | 82 | 72 | 75 | 65 |
  | ours, 10,000 traversals: policy net | | 202 | 113 | 95 | 78 |
  | ours, 10,000 traversals: SD-CFR average | 367 | 199 | | | |
  | ours, 10,000 traversals, `--masked-loss`: policy net (stopped at 300) | | 206 | 129 | 101 | |
  | ours, 25,000 traversals: SD-CFR average | 178 | 121 | | | |
  | ours, 25,000 traversals: policy net | | 117 | 91 | 69 | 75 |
  | ours, 25,000 traversals, batch 40,000 + cosine from 3e-3: SD-CFR average | 234 | 134 | | | |

  The 450-iteration policy net (77.7, exact) is where the runs on the three-bet flop with the earlier network ended
  (80-86): correcting the game, the network, the loss weights and the head initialisation did not move the floor.

  What was measured about the gap:
  - *Traversals.* With 25,000 traversals - the paper's nodes touched per iteration - the average is at 178 after 50
    iterations (paper 154); with 10,000 it is at 367. The paper's K = 10,000 is therefore not our `--traversals 10000`.
  - *Where the error is.* Of the policy net's 202 at t = 100, deviations on the pre-flop street alone are worth 49 and
    deviations on the flop alone 145 (`holdem_br --br-streets`).
  - *Refit noise.* Two consecutive regret nets (iterations 99 and 100, 98 % the same data) disagree by 15-19 chips rms
    at the 1,352 pre-flop infosets, where the predicted regrets have an rms of 75-82; 90 % of the per-iterate
    strategies there are pure, and consecutive iterates differ by 0.26 (small blind) and 0.49 (big blind) in L1
    (`docs/analysis/fhp_preflop_refit_noise.py`). Regret matching turns the fit's noise into the strategy.
  - *The fit.* Refitting one seat's memory at t = 125 under different recipes (`docs/analysis/fhp_fit_noise.py`; per-net
    noise at the pre-flop infosets in chips, where the tabular mean of the same samples has a standard error of 4.5;
    held-out loss relative to the paper's recipe):

    | recipe (4,000 steps of 10,000 unless stated) | noise | distance to the tabular mean | held-out loss |
    |---|---|---|---|
    | the paper's: constant 1e-3, last weights | 6.7 | 7.2 | 1 |
    | loss weights raw instead of 2/T; zero instead of random head | 7.1-7.6 | 7.2-8.4 | 1.000-1.003 |
    | `--net paper` (the network before the audit) | 11.0 | 9.4 | 1.042 |
    | `--weight-average 0.998` | 2.7 | 4.9 | 0.987 |
    | `--lr-schedule cosine` | 2.9 | 4.8 | 0.992 |
    | batch 40,000 | 5.0 | 5.8 | 0.960 |
    | batch 40,000, cosine from 3e-3 | 2.4 | 4.3 | 0.937 |
    | 8,000 steps, cosine | 2.5 | 4.3 | 0.956 |
    | 16,000 steps | 4.4 | 5.5 | 0.948 |
    | 16,000 steps, weight average | 2.2 | 4.1 | 0.931 |

    The 4,000-step fit both underfits (16,000 steps take 5 % off the held-out loss, of which most is irreducible
    sampling noise) and jitters; neither the 2/T weights nor the head initialisation matter, and the audit's network
    is better than the one before. This agrees with the paper's own SGD-step sweep (more steps, lower floor).
    `--batch-size 40000 --lr 3e-3 --lr-schedule cosine` gets most of the 16,000-step fit at 1.1-1.6 times the cost of
    the paper's.
  - *The tighter fit in training.* The benchmark is one memory at t = 125. A 100-iteration run with that fit and
    25,000 traversals (`runs/fhp4_k25_fit`) is worse early: 234 / 134 at t = 50 / 100 against 178 / 121 with the paper's fit. At
    iteration 1 the memory holds ~125,000 samples and 4,000 batches of 40,000 pass over it 1,280 times; the fit then
    reproduces the sampled regrets, noise included. The runs of September with 16,000 steps (three-bet game) showed
    the same shape - 276 / 145 / 87 at t = 50 / 100 / 200 against 179 / 116 / 96 - i.e. a tighter fit pays only once
    the memories are large.
  - *The traversal count, cross-checked.* The DREAM paper (Steinberger, Lerer, Brown) runs this setup with public code:
    10,000 external-sampling traversals, 4,000 batches of 10,000 (`FHP_ES.py`, `HYPERS.py`). Its FHP figure advances
    by 0.245e8 nodes per 60 iterations, i.e. 4.1e5 nodes per iteration - ours (4.1e5 under uniform play). So our
    traversals are the usual ones, and the Deep CFR paper's run saw 2.5 times more per iteration.
  - *The floor does not depend on the traversals.* 25,000 traversals end at 75 after 450 iterations (69 at 300),
    10,000 at 78: "the same final exploitability" as in the paper - but at the level of its 4,000-step sweep curve.
  - *The paper's default is not its 4,000-step curve.* In Fig. 3-middle 4,000 steps end at 65 (1,000: 110, 8,000: 43,
    16,000: 37, 32,000: 35, all started with many traversals: 230-420 at t = 10). The "Deep CFR (5 replicates)" of
    Fig. 4 and the large-K curves of Fig. 3-left, nominally the same configuration, end at 34-40: they coincide with
    the sweep's 16,000-32,000-step curves. The text speaks of "reducing the number of SGD steps" from the default.
  - *The fit budget moves our floor the same way.* The 25,000-traversal run, continued from its checkpoint at
    t = 325 to 450 with another fit (exact total; street splits on 2,000 flops):

    | fit of the last 125 iterations | total at t = 450 | pre-flop deviations only | flop deviations only |
    |---|---|---|---|
    | 1,000 steps | 90.3 | 29.8 | 64.3 |
    | 4,000 steps (the run itself) | 74.8 | 30.1 | 47.7 |
    | 4,000 steps, `--weight-average 0.998` | 72.7 | 20.3 | 52.3 |
    | 8,000 steps | 60.8 | 22.6 | 42.4 |
    | 16,000 steps | 58.1 | 22.6 | 39.1 |

    The pre-flop part stays at 26-31 from iteration 200 on in every 4,000-step run, with 10,000 or 25,000 traversals;
    weight averaging (less refit noise) lowers it by a third and more steps lower both parts.
  - *Pre-flop is limited by the fit, not by the data.* The memories hold thousands of samples per pre-flop infoset, so
    the mean regret can be tabulated (`docs/analysis/fhp_preflop_tabulation.py`). At t = 450 its standard error is
    2.2-3.1 chips; the last network is 5.9-8.9 chips away from it and two consecutive networks 6.4-8.9 chips from
    each other, while the two best actions are less than 10 chips apart in 52-79 % of the infosets: regret matching
    on the network picks another most-likely action than the tabulated regrets in 15-17 % of them. The policy network
    reproduces the tabulated average strategy to 0.03 in L1.
  - *Ruled out.* Suit symmetry: single iterates are not symmetric (isomorphic infosets differ by 0.12-0.15 in L1), but
    a network fed suit-canonical cards fits no better (held-out loss 39,746 against 39,398;
    `docs/analysis/fhp_canonical_suits_fit.py`).
  - *16,000 steps from scratch do not reach the paper.* `runs/fhp4_k25_steps16k`: 136 / 87 / 68 at t = 100 / 200 / 300
    (4,000 steps: 117 / 91 / 69; the paper's 16,000-step curve: 55 / 42 / 42). More steps pay late - the 4,000-step run
    rises again from 69 to 75 between t = 300 and 450, the branches fall to 58-61 - but not to 37.
  - *An independent re-implementation is in the same place.* Liu et al. (Neural ReCFR-B, arXiv 2012.01870) implement
    Deep CFR on OpenSpiel "with the hyper-parameters given in" the paper (10,000 traversals, 4,000 steps of 10,000,
    40 M memories) and report 47.0 mbb/g after 6.0e8 nodes touched, rising to 73.0 at 1.3e9. The paper's curve is at 37-40
    after 3.3-4.3e8 nodes. By nodes our 10,000-traversal run is at 78 after 1.9e8 nodes and falling like t^-0.5
    (113 / 95 / 78 at t = 200 / 300 / 450), which extrapolates to their value; `runs/fhp4_paper_long` continues it to
    1,500 iterations (6e8 nodes) with 40 M advantage memories.
  - *What the remaining error is not.* Memory size: 6 M instead of 24 M memories give 128 / 97 instead of 117 / 91 at
    t = 100 / 200 (`runs/fhp4_k25_mem6m`). Independent noise of a run: the mixture of the t = 300 policies of two
    independent runs (69.2 and 67.9) has 63.7 (`docs/analysis/fhp_ensemble_br.py`) - the weaknesses are shared. The
    card embeddings: without the per-card table, or with twice the width, the fit changes by < 0.5 % in held-out loss.
    Rare situations: with deviations at one kind of decision node only (`holdem_br`'s `br_filter`,
    `docs/analysis/fhp_br_by_node.py`), the 75 of the 25,000-traversal run come from the most frequent decisions -
    the small blind's first action 15.1, the big blind facing a raise 11.1, the first flop action 12.9, facing a flop
    bet 10.5 / 9.5, after a flop check 7.6 - and next to nothing from re-raised pots (0-3 each).
  - *Open.* The precision of the regret network at the best-sampled decisions: 8,000 steps with weight averaging from
    scratch (`runs/fhp4_k25_8k_ema`), and the long run above.

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
| PCFR+: the prediction is the last instantaneous regret, as in Xu et al.'s PDCFR+ code that our DCFR+ / PDCFR+ follow, not Farina et al.'s bottom-up construction (Leduc at 1000 iterations: 7.1e-4 against 3.3e-4). | **by choice** - one prediction rule for the whole predictive family |
| Small-game MCCFR pruning: the 95 % draw was per node, without Pluribus's exemptions for the last round and terminal-leading actions, and without a regret floor. | **fixed** - `test_mccfr_pruning_follows_the_pluribus_rule` |

## Deep (Predictive) Discounted CFR (`headsup.algos.pdcfr`)

Written against the authors' code; the deviations from the paper's text and from the code are listed in
`docs/superpowers/specs/2026-10-07-deep-pdcfr-design.md`. An independent review compared the sampler, the strategy
rule, both fits and the average-policy fit with the authors' code (120k Leduc episodes, 20,000 random strategy cases,
losses equal to 15 digits) and found no difference.

## ReBeL (`headsup.algos.rebel`, Leduc and Kuhn)

Written against the official Liar's Dice code (facebookresearch/rebel) where the paper leaves a choice open, and
against the paper's poker appendix and the authors' answers (issue #20) for the board card, which Liar's Dice does not
have. `docs/superpowers/specs/2026-10-08-rebel-design.md` has the table of paper-vs-code differences and the list of
deviations. In short:

| item | status |
|---|---|
| Stopping step: uniform over all steps in self-play, the average's weights (even steps, weight j + 1) at test time; "1024 iterations" = 512 updates per player; regrets and ranges smoothed with 1e-80; the loss is twice the textbook Huber. | **by choice** - the official code, where the paper's text and its two algorithm listings disagree; `--train-stop`, `--regret-floor` |
| Card removal, which the Liar's Dice code does not need: the self-play walk draws the action from the actor's hand given both ranges, the board card comes from the PBS, values before the card average the values after it with the surviving opponent mass. | **by choice** - exact against enumeration (`test_ranges_after_actions_and_the_board_card_match_brute_force`, `test_profile_values_match_the_game`) |
| One root solve per network version instead of one per playthrough (the root PBS is fixed, CFR is deterministic); every reached end-of-round PBS becomes an example (the paper: one in three). | **by choice** - the same distribution, less compute |
| Minibatches per epoch and learning rate (200, 1e-3 halved every 100 epochs; official Liar's Dice: 50, 3e-4 halved every 400). | **by choice** - measured on Leduc; the official values are flags |
| Evaluation: exact exploitability of the policy played in expectation, next to the paper's average of 1 024 sampled playthroughs. | checked - the sampled number is an upper bound in expectation (measured: K = 16 / 64 / 256 / 1024 = 244 / 158 / 136 / 129 against 127 exact) |
| No Leduc result is published. Search with exact leaf values reaches 22.1 mA/g at 1024 steps (unsafe search: 54.5). | checked (`python -m headsup.algos.rebel --game leduc --oracle`) |
| CFR-AVG, the policy network and warm start, subgames cut after a number of actions, hold'em. | **open** - not implemented |

## Pluribus-style blueprint, search and LBR

Consistent with the paper: the MCCFR-P update rule, regret matching, the shape of the pruning rule and the floor ratio,
the linear discount, external sampling, blueprint play (average strategy), the search root at the start of the round,
the belief updates, final-iterate play in Pluribus mode, the vector Linear CFR, DCFR's constants, and LBR's action
values (equal to the paper's Fig. 1 formula to 5.5e-14).

| item | status |
|---|---|
| Bucket assignment was stochastic: the flop / turn bucket of a (hand, board) came from a fresh Monte-Carlo estimate at every lookup (standard deviation 5 of 200 buckets, modal bucket 16 % of the time), and 1326-hand queries used another estimator. The paper puts every situation into one bucket. | **fixed** - the estimate is seeded by the cards; the exact per-board table abstraction (`--abstraction table`, potential-aware k-means as in the paper) uses all 1,081 completions of a flop; `test_bucket_of_a_situation_is_a_function_of_the_cards` |
| Search re-solved at every decision of a round and froze its own earlier action as a one-hot row, which inflated the real hand's weight in its own range by 1 / sigma(a) (8.5 % of post-flop rounds). Algorithm 2 searches when a round begins. | **fixed** - one solve per betting round, every decision reads it; `test_pluribus_mode_solves_once_per_round_with_public_beliefs` |
| The opponent's belief started conditioned on the searcher's hole cards; the paper keeps beliefs "from an outside observer's perspective" (1/1326 each). | **fixed** - public beliefs in Pluribus mode (same test) |
| Later-round bucket regrets in the vector solver ignored the solving player's own range weights (the subgame's root deals hands in proportion to their reach). | **fixed** - `test_bucket_regrets_are_weighted_by_the_solving_players_range` |
| LBR against a search player queried its blueprint, not the searched strategy (total-variation distance 0.40 between the two). | **fixed** - the player answers for every hand from its round solve (`all_hands_probs`), and LBR asks it; `test_lbr_measures_the_searched_strategy_of_a_pluribus_player` |
| Pruning never fired at our budgets: the threshold (-3e6) was scaled by the stack only; after 20 M iterations the most negative regret is -1.39e6. | **fixed** - the threshold is -scale x stack x iterations (`--prune-scale`); every chunk reports the share of regrets below it |
| Average strategy: reach-weighted at every sampled opponent infoset on every round, not Algorithm 1's round-1 counters plus later-round snapshots. | **by choice** - better in our measurement (508 / 541 against 629 / 663 mbb/g with the counters every 10,000 iterations); `--average counters --strategy-every N` selects the counters |
| Linear-CFR / pruning schedule: 40 % / 20 % of the run against Pluribus's 3.5 % / 1.7 %. | **by choice** for runs of tens of millions of iterations (the first 40 % are down-weighted); the help text gives Pluribus's proportions |
| Abstraction: 200 buckets per round by expected hand strength (`mc`) or by (mean, std) of the equity over the completions (`table`); Pluribus clusters equity distributions with k-means. | **by choice** - `table` is the closer one |
| Depth-limited mode (`search:<net>` without `@pluribus`): one sampled continuation by default, where both papers warn about fixed leaf values and let the players choose among four (`@leaf4`); it plays the average rather than the final iterate; `@leaf4` cannot be used with a tabular blueprint. | **open** - documented in `headsup/search.py`; the default is unchanged until it has been measured |
| LBR: flop equities from 200 Monte-Carlo runouts (paper: exhaustive); no round-restricted variants (the paper's Table 2). | **fixed** - the flop's 1,081 runouts are enumerated by default (`--max-exact 100` = the old sampling); `--from-round flop` etc. = LBR that only check/calls before that round. Pre-flop equities stay sampled (`--mc-samples`). |

## AlphaHoldem (`headsup.alphaholdem`)

There is no official code and the paper leaves most of the training procedure open; every open point, the choice
made and the reason are in `docs/superpowers/specs/2026-10-08-alphaholdem-design.md` (22 rows). As the paper states
them: the two tensors (card 6 x 4 x 13; action 24 x 4 x nb with the rows player 1, player 2, sum, legal-at-that-
decision), two ConvNets without shared parameters, gamma 0.999, GAE lambda 0.95, delta1 3, Adam 3e-4, minibatch
16,384, 131,072 decisions per iteration, one main agent against a pool of its K best snapshots by ELO.

| item | status |
|---|---|
| The game: 4 actions (one check/call column), 50 bb, this repository's raise cap, against the paper's 9 actions and 200 bb. | **by choice** - the shipped baselines exist only in the default tree; a larger tree only changes tensor sizes |
| Eq. (3) as printed has no `min` and no sign condition. Implemented: PPO's clipped surrogate, for A < 0 not below delta1 x A (the dual-clip loss the text cites; for A < 0 it equals the printed formula). | **by choice** - `test_trinal_clip_policy_objective_by_hand` (also shows that a `min` over three terms is PPO again) |
| Value-clip bounds delta2 / delta3 ("the total number of chips the player has placed and the opponent has placed"). Read per hand, a complete hand's return never leaves them, so with our Monte-Carlo returns the value target is the return: the default. Read per state (the chips put in up to the state; OpenHoldem's "state value when the player / the opponent folds") the bounds are asymmetric when facing a bet, the clipped target's mean lies above the expected return (7.7 - 9 chips at near-uniform play) and with lambda 0.95 the advantages of decisions followed by another one of the same seat are biased upwards (+0.95 / +0.50 / +0.27 chips over iterations 0-20 / 20-50 / 50-90; 0 without the clip - measured in the review of 2026-10-08). | **by choice** - no clip; `--value-clip` is the per-state reading; `test_per_state_value_clip_biases_the_advantages_of_earlier_decisions` shows the mechanism. Its effect on playing strength is **open** (two GPU runs compare the variants) |
| Rollouts hold complete hands, the value target is the hand's discounted return, no bootstrap; the paper has 128-step rollouts. | **by choice** |
| One process, synchronous on-policy PPO, a network of 1.4 M parameters (paper: 8 GPUs, "off-policy ... replay buffer", 8.6 M). | **by choice** |
| Epochs, loss coefficients, advantage normalisation, gradient clipping, K, snapshot interval, opponent sampling, ELO from blocks of hands: not in the paper. | **by choice** (PPO defaults and the brief's values), none tuned |
| No finished long run yet. After 0.1e9 samples (the paper's ablations: 0.65e9; two GPU runs, one seed each): +0.29 ± 0.05 chips/hand against the DeepCFR net, −0.66 ± 0.04 against the blueprint, LBR 3.62 ± 0.17 without the value clip; +0.10 ± 0.05 / −1.10 ± 0.05 / 3.45 ± 0.15 with the per-state clip. The paper's results (Slumbot, human matches) are in another game and cannot be compared. | **open** |
