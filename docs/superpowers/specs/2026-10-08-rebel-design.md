# ReBeL on the small poker games - design

Status: implemented on 2026-10-08 under a standing approval for autonomous work; this spec records the design as
built. An independent review found no correctness bug (brute-force best responses against the explicit mixture
agree with the repository's to 1e-16) and its findings - test margins, reproducibility across thread counts, edge
cases - are worked in. Validation, three seeds with the default settings, 300 epochs: 27.7 +- 1.4 mA/g for the
policy played, 28.2 +- 1.3 for 1 024 sampled playthroughs, 53.3 +- 2.0 for unsafe search, against 22.1 / 24.8 /
54.5 for the search with exact leaf values: 1.25 times the exact-leaf search, criterion 2 asks for at most 2.

Sources: Brown, Bakhtin, Lerer, Gong, "Combining Deep Reinforcement Learning and Search for Imperfect-Information
Games", arXiv 2007.13544 (NeurIPS 2020), TeX source with all appendices; the official Liar's Dice code
github.com/facebookresearch/rebel @ 7960a42 (`csrc/liars_dice/subgame_solving.cc`, `recursive_solving.cc`,
`recursive_eval.cc`, `cfvpy/models.py`, `cfvpy/selfplay.py`, `conf/c02_selfplay/liars_sp.yaml`) and the authors'
answers in its GitHub issues (#10, #17, #20, #26); the research brief of 2026-10-07 with its scratch prototype.

## Goal

ReBeL - self-play reinforcement learning plus search over public belief states - for Leduc hold'em, as a
self-contained solver (`headsup/algos/rebel.py`) whose test-time policy is evaluated by exact best response. Kuhn
runs through the same code as a smoke test: it has one betting round, so its only subgame is the whole game and no
network is involved.

Non-goals: FHP and no-limit (stages ii and iii of the brief), CFR-AVG and the policy network / warm start (the
paper's poker agent; the official repository has neither), subgames cut after a number of actions (the official
Liar's Dice setting; a betting round of Leduc is small enough to solve whole), a player for the hold'em engine.

There is no published Leduc result of ReBeL: the authors chose Liar's Dice instead (NeurIPS author feedback). The
references are a calibration with exact leaf values and two third-party numbers (see "Success criteria").

## Algorithm

**Public belief state (PBS).** The public state - pot, board card, whether the round is over - plus one range per
player: a probability vector over its six possible private cards. The betting history is not part of it.

- After a public action `a` of player `i` with policy `pi`: `range_i(h) <- range_i(h) pi(a | h)`, normalised; the
  other range is untouched.
- Normalisation adds `1e-80` to every entry, as the official code does: a range without any mass becomes uniform.
- The board card is a public chance event: only the public state changes. The hand the card blocks is set to
  zero in both ranges and they are renormalised. A PBS stands for the joint distribution
  `range_0(h0) range_1(h1) [h0 != h1]`; this update keeps it exact (test against brute force).
- The card itself is drawn from the PBS: `P(c)` is proportional to the mass of the hand pairs that do not contain
  `c`. It is not uniform.

**Subgame.** One betting round from a PBS. Leduc's first round has 6 public decision states, 4 fold terminals and
5 leaves (check-check, check-raise-call, check-raise-raise-call, raise-call, raise-raise-call; pots 2 / 6 / 10 / 6
/ 10): the public states where the round is over and the board card is due. The last round has the same shape
with showdowns in place of the leaves, so its solve needs no network. Terminal values are exact: against the
opponent's reach, with card removal (the pair of hands that share a card has no mass; the hand the board blocks has
value 0).

**Value network.** `v(PBS, agent)` returns one value per private card of the agent: the expected payoff of that
hand in chips against the opponent's *normalised* range (pairs sharing a card contribute nothing), whatever the
agent's own range gives the hand. It becomes a counterfactual value by multiplying with the opponent's total reach
of the leaf. One network serves all PBS kinds, one agent per query; there is no zero-sum correction (as in the
paper and the code).

**Search: Linear CFR-D** (the official `CFR` solver with `linear_update`), vectorised over hands and over
subgames. One step updates ONE player `p` (the traverser); `T` steps are `T / 2` updates per player:

1. Both players' reaches of every node from the root ranges under the current iterate `sigma^t` (uniform at
   `t = 0`).
2. Leaves: the network is asked AGAIN, for the traverser, at the PBSs the CURRENT iterate leads to (both reaches
   normalised): `cfv_p(leaf, .) = v(pub(leaf), norm(reach_0), norm(reach_1), agent = p) * sum(reach_{1-p})`.
   Terminals: exact payoffs against `reach_{1-p}`.
3. Backward pass; `regret_p += cfv_p(child) - cfv_p(node)`.
4. `root_mean_p <- root_mean_p + (cfv_p(root) - root_mean_p) * 2 / (k + 2)` for the traverser's `k`-th update
   (`k = 0, 1, ...`): the linear average of the iterates' root values.
5. New strategy of `p` by regret matching on `max(regret, 1e-80)`; then `regret_p` and `strategy_sum_p` are
   multiplied by `(k + 1) / (k + 2)` and `strategy_sum_p += own reach * new strategy`. The uniform strategy is
   in the sum from the start with weight 1, so the iterate after `j` updates has weight `j + 1`.

The leaf values belong to the iterate, not to the average: what the average policy is worth at a leaf is not what
the search optimised against.

**Why a random stopping step.** CFR's guarantee is for the average strategy, but the continuation after a leaf
was valued - and at test time is solved - for the ranges of single iterates. Playing the average strategy and
handing its ranges to the next subgame ("unsafe" search) is exploitable: the paper's example is modified
rock-paper-scissors, where it ends in a pure strategy. ReBeL instead stops the solve at a random step `t*`,
plays the iterate of that step for the whole subgame and hands down that iterate's ranges for both players. With
`t*` drawn from the weights the iterates have in the average, the policy played is the average in expectation
(Theorem 3), while every single playthrough is consistent with the ranges it uses. Measured here with exact leaf
values at T = 1024: 22.1 mA/g for the random-iterate policy against 54.5 for unsafe search.

**Self-play** (one playthrough; the paper's Algorithm 1 as the official `RlRunner::step` runs it):

1. Solve the round from the current PBS for `T` steps with the network at the leaves. Store the example
   `(PBS, agent, root_mean_agent)` for both agents.
2. Draw the stopping step `t*`; one player of the two is the explorer.
3. Walk the iterate of step `t*` from the root to a leaf or terminal: at each decision the explorer takes a
   uniformly random legal action with probability `eps = 0.25`; otherwise the action is drawn from the iterate
   for a hand drawn from the actor's range. After EVERY action, explored or not, the actor's range is updated
   with the ITERATE's probabilities.
4. At a terminal the playthrough ends. At a leaf: the PBS there gets an example (below), the board card is drawn,
   and the last round is solved from the resulting PBS - `T` steps, no network - which gives the example
   `(PBS after the card, agent, root_mean_agent)`. The last round ends in terminals, so nothing follows.

The stored target is the solver's weighted average of the iterates' root values over all `T` steps - not the value
of the average strategy, not the value of the iterate that was walked. No game outcome is used.

**Targets before the board card** (`--leaf-targets`). The search asks the network about PBSs at the END of the
first round, but a solve starts AFTER the card. Two ways to make the missing examples:

- `net` (default; the paper's poker agent, described by its authors in issue #20): for a PBS reached at the end of
  the round the network is asked for the values after every possible card, and their average with card removal
  is the example:
  `v(h) = 1 / (C - 2) * sum over c != h of mass_opp(c) * v_c(h)`, `mass_opp(c)` = the opponent's mass that
  survives `c`, `v_c` = the values at the PBS with board `c` and both ranges renormalised without `c`, `C = 6`.
  The network learns both kinds: values after the card from the exact last-round solves, values before it from
  itself.
- `solve`: the last round is solved for all six cards and the same average of the solves' `root_mean` is the
  example. Exact up to the solver's convergence, six solves per example; only possible because Leduc has six
  cards. No examples are stored after the card (nothing would read them).

**Value network and training.** Input, 21 numbers for Leduc: agent index (1), a flag "the round is over and the
board card is due" (1), pot / 26 (1), the board card one-hot (6, zeros before it), both ranges (2 x 6). Output:
6 values in units of the most a player can win (13 chips; the largest pot is 26, which is what the pot feature
is divided by), so targets lie in [-1, 1]. Architecture as the official `Net2`:
`[Linear -> LayerNorm -> GELU] x 2` with 256 units and a linear output layer scaled by 0.01. Loss: the official
"Huber", `x^2` for `|x| <= 1` and `2 |x| - 1` beyond, averaged over hands and examples (with targets in [-1, 1]
it is the squared error in practice). Adam, gradient-norm clipping at 5, minibatches of 512 drawn uniformly from
a FIFO buffer of 2 M examples.

**One epoch**: a root solve with the current network, 2048 playthroughs below it (about 7 000 new examples), 200
minibatches. Learning rate 1e-3, halved every 100 epochs at most twice; a run is 300 epochs = 60 000 minibatches.
These three numbers are the only training hyperparameters not taken from the official Liar's Dice configuration
(50 minibatches per epoch, 3e-4 halved every 400 epochs, 1 000 epochs = 50 000 minibatches); see "Deliberate
deviations" 9 for the measurement behind them.

**Test-time play** (`Playthrough`, `policy(state) -> probabilities`): the same search without exploration. The
root solve is stopped at an even step `t* = 2 j` drawn with probability proportional to `j + 1` (`j < T / 2`);
that iterate is played throughout the first round. After the card the last round is solved from the ranges that
iterate gives BOTH players at the leaf, stopped at a step drawn the same way, and that iterate is played to the
end. One stopping step per subgame, kept for all its decisions.

**Evaluation** (`evaluate()`; all numbers are exact best responses over the full game tree, in chips = antes, mean
over the two seats):

| key | what it is |
|---|---|
| `exploitability` | The exploitability of the policy ReBeL plays, taken as the mixed strategy over its own random stopping steps. Exact, not a bound: the mixture over the last round's stopping steps is that solve's average strategy after `T - 2` steps, so the policy is the mixture over the `T / 2` root iterates of [iterate `j`, the last-round averages at iterate `j`'s ranges]; all `T / 2 x 5 x 6` last-round solves run in one batch. |
| `exploitability_sampled` | The paper's protocol: the exploitability of the average of `K` sampled playthrough policies (`K = 1024`). In expectation an upper bound of `exploitability`, because a best response's value is convex in the policy it responds to; a single draw can fall below it. It costs nothing extra: the stopping steps are snapshots of the same batch. |
| `exploitability_unsafe` | Unsafe search for comparison: the root's average strategy and the last round solved at the average's ranges. |
| `value_error` | RMS error of the network in chips on a fixed probe set: the leaf PBSs of the even iterates of a 64-step search with exact leaf values (weights `j + 1`), each solved to the end for `T` steps. |
| `value_error_search` | RMS error in chips of the leaf values the evaluated search was actually fed (network output x opponent reach) at its own iterates' leaf PBSs, against the exact solves of the same batch. |
| `root_value` | Player 0's game value according to the root solve's `root_mean` (exact: -0.0856). |

How fast the sampled mixture approaches the exact one (exact leaf values, T = 64, 12 draws each): K = 16:
244 +- 14, K = 64: 158 +- 5, K = 256: 136.0 +- 2.4, K = 1024: 128.9 +- 1.0 mA/g against 126.8 exact; the
standard deviation of a single K = 1024 draw is 3.5 mA/g.

`--oracle` runs the search with exact leaf values (every leaf query solves the rest of the game) and evaluates it:
the calibration of what the search alone can reach.

## Where the paper and the official code disagree

The default follows the code: it produced the published Liar's Dice numbers. The two rows marked (*) are the
exception: the official values are a flag away, the defaults were measured on Leduc (deviation 9).

| detail | paper | official code = default | switch |
|---|---|---|---|
| stopping step in self-play | Alg. 1: uniform on `t_warm + 1 .. T`; Sec. 5.2: uniform on `0 .. T - 1`; Alg. 2 (Linear CFR-D): `P(t)` proportional to `t` | uniform integer on `0 .. T`, both ends and the odd steps included, also with linear weights | `--train-stop uniform` / `linear` (= the test-time weights) |
| stopping step at test time | "a random iteration" | even steps `2 j < T` with weight `j + 1`; the solver stops there | - (the theorem needs the average's weights) |
| what "1024 iterations" are | iterations | 1024 single-player steps = 512 updates per player | `--iters` counts steps |
| average of root values (the training target) | Alg. 1: plain mean; Alg. 2: linear | linear, step `2 / (k + 2)`, per player over its own updates | - |
| first iterate | uniform `pi^0` | uniform and already inside the average with weight 1 | - |
| regret matching | not specified | regrets floored at `1e-80`: an action without positive regret keeps a probability of that order | `ReBeL(regret_floor=0)` / `--regret-floor 0`: textbook rule |
| a range without mass | not specified | becomes uniform (`1e-80` added before normalising) | - |
| hand of the walk | sampled once at the root | redrawn from the range at every decision (the same distribution) | - |
| value-net input | poker: agent index, acting agent, pot / stack, board cards, ranges | Liar's Dice: acting player, agent index, last bid, ranges | ours: see "Deliberate deviations" 5 |
| loss | "pointwise Huber" | `x^2` / `2 |x| - 1`: twice the textbook Huber | - |
| gradient clipping | not mentioned | norm 5 | `--grad-clip` |
| learning rate (*) | Liar's Dice: 3e-4, halved every 400 epochs; TEH: 4e-4, halved every 100 epochs | 3e-4, halved every 400 epochs, at most twice | default 1e-3, halved every 100 epochs at most twice; `--lr`, `--lr-halve-every` |
| replay | Liar's Dice: not stated; poker: 12 M circular, half removed after 20 epochs | 2 M FIFO, uniform | `--buffer` |
| training per epoch (*) | 25 600 examples sampled = 50 minibatches of 512; 1 000 epochs | the same; at least one new example per four sampled (`train_gen_ratio`), network sent to the generators every epoch | default 200 minibatches per 2 048 playthroughs (one new example per 14 sampled), 300 epochs; `--games`, `--steps`, `--epochs` |
| examples before a chance card | not in the paper | not in the code (Liar's Dice has no public chance); issue #20: one in three of the reached end-of-round PBSs, the network's values after all next cards averaged "accounting for blockers" | `--chance-prob` (default 1), `--leaf-targets` |
| evaluation | average of 1 024 sampled playthrough policies, "upper bounds" | the same (`recursive_eval.cc`); its monitor during training uses the unsafe average-policy recursion | all three numbers are reported |
| leaf values at poker's chance nodes | "solves to the end of the current betting round": leaves before the card | - | as the paper |

## Deliberate deviations

1. **One root solve per epoch.** The first round's root is always the same PBS and CFR is deterministic, so every
   playthrough of a network version would repeat the same solve. It runs once and all playthroughs draw their
   stopping step from its recorded iterates: the same distribution as the official loop with the network updated
   once per epoch. The root PBS's example is stored once per solve instead of once per playthrough.
2. **Card removal in the walk.** The official walk draws the actor's hand from its range alone, which is right for
   Liar's Dice. In poker the two hands exclude each other; the action is drawn from the hand distribution given
   both ranges, `range_j(h) (1 - range_opp(h))` up to normalisation, so a public line has exactly the probability
   the PBS gives it.
3. **Every reached end-of-round PBS becomes an example** (`--chance-prob 1`); the paper's poker agent takes one in
   three to save network queries over 22 100 flops. Leduc has six cards.
4. **`--leaf-targets solve`** does not exist in the paper (see above). It is the cheap way to separate the
   search from the bootstrapping; `net` stays the default.
5. **Network input.** No "acting agent" input: player 0 opens every round, so it is constant at every PBS the
   network sees. A flag separates the end of the first round from its start (the same pot, no board card; the
   paper's stated input cannot tell them apart). The board card is a one-hot, not an embedding.
6. **Exact evaluation of the played policy** in addition to the paper's sampled one (possible because the last
   round's mixture collapses to an average strategy).
7. **Evaluations draw their stopping steps from a generator of their own**, so a run does not depend on how often
   it is evaluated.
8. **No burn-in, no half-buffer purge**: the first epoch already holds 14 minibatches of data; the purge is the
   paper's poker setting and is not used for Liar's Dice either.
9. **More training per generated example, a higher learning rate.** The official throttle (one new example per
   four sampled, 50 minibatches per epoch) fits a setting where the search is the expensive part. Here an
   epoch's playthroughs cost about 2.5 s and 50 minibatches 0.25 s, and the network underfits its buffer: 6 000
   more minibatches on a frozen buffer still lowered the loss. With the official schedule (3e-4, 50 minibatches)
   seed 0 reached 63 mA/g after 120 epochs (7 min) and stayed at about 40 from epoch 300 (19 min) on; with 200
   minibatches and 1e-3 it was at 40 after 35 epochs (4.5 min) and at about 30 once the rate had been halved
   (epoch 120, 12 min). A 3 x 512 network was no better per epoch at twice the cost. The official values:
   `--steps 50 --lr 3e-4 --lr-halve-every 400 --epochs 1000`.
10. **Not implemented**: CFR-AVG, fictitious play, the policy network, depth limits in actions.

Inherited from the official code rather than a deviation: the self-play walk normalises a range after every
action, while the leaf query and test-time play normalise the product of the reaches once at the leaf. The two
differ only on lines where a range loses all its mass on the way (the walk restarts it as uniform there and keeps
updating it; the product stays without mass and becomes uniform at the leaf): 0 of 13 073 leaf visits with the
trained network.

**Reproducibility.** A seed fixes a run only together with the number of torch threads: the thread count changes
the order of the network's float additions, and the search amplifies the last digit (the same seed under 1 and 2
threads: the root iterates differ from the first step on, the first evaluation gives 1.28 against 1.03). The CLI
therefore runs with `--threads 1` unless told otherwise (12 % slower per epoch than 2 threads). Runs are compared
as distributions over seeds, never seed against seed.

Known and deferred (minor): `value_error_search` compares the network with solves of `T - 2` steps where the
training targets come from `T` steps; resuming a checkpoint checks the game, the network size and the buffer
capacity but no other hyperparameter.

## Architecture

- `headsup/algos/rebel.py` (new)
  - Ranges: `normalise`, `bayes_update`, `board_distribution`, `sample_board`, `deal_board`, `chance_average`,
    `stop_weights`.
  - `RoundTree(state)`: the public tree of a betting round from the game protocol (node kinds, children, stakes
    read off `returns()`, betting histories for the info keys) with the index arrays for vectorised reaches.
  - `CFRD(tree, beliefs, show, amount, mask, leaf_fn)`: the solver for `B` subgames at once. `leaf_fn(p, beliefs)`
    is the only door for leaf values: the network, an exact solve, or a test stub. Internally the subgame axis is
    the last one, so every numpy operation of a step runs over whole blocks of memory.
  - `sample_leaf`: the walk of step 3 for all playthroughs at once.
  - `value_net`, `huber`, `Replay`.
  - `ReBeL(game, ...)`: `search()`, `self_play()`, `generate()`, `train()`, `iterate()`, `policies()`,
    `playthrough()`, `evaluate()`, `probe()`, `state_dict()` / `load_state_dict()`; `Playthrough`.
  - CLI: `python -m headsup.algos.rebel --game leduc --epochs N --seed S --json out.json --checkpoint ck.pt
    [--device cpu|cuda:0] [--threads 1]`, `--oracle` for the calibration. The curve file is replaced atomically
    at every evaluation. The JSON curve has per evaluation: `epoch`,
    `examples`, `sgd_steps`, `games`, the six evaluation keys above with `samples`, `loss`, `seconds`.
- `headsup/algos/leduc_report.py`: a ReBeL section - the reference rows, each labelled with its source and what
  it measures, and our runs `runs/leduc_rebel/leduc_s*.json` by epoch (mean +- sd over seeds).
- Restricted to the Leduc family (`headsup/games/leduc.py`): one private card per player, at most two betting
  rounds, a last-round tree that does not depend on the first round's betting. The constructor asserts it.

## Tests (`tests/test_rebel.py`, 48 tests, 15-20 s on CPU; the report's test is in `tests/test_deep_algos.py`)

A convergence test cannot tell most one-line mistakes apart, so the pieces are tested directly:

- Trees against the games (node counts, pots, stakes, histories); a player's reach as the product of its action
  probabilities.
- Ranges after a betting line and after the board card, and the card's distribution, against the joint
  distribution of (hands, betting, card) enumerated with the game.
- The root values of a random profile through the round trees (terminals with card removal, leaf value x
  opponent reach, chance average, info keys) against `expected_value` of the game: equal to 1e-12, Leduc and Kuhn.
- The solver on Kuhn with textbook regret matching equals the repo's tabular Linear CFR (average strategy after
  `n` updates per player = tabular after `n + 1` iterations) to 1e-9; the official `1e-80` floor separately.
- `root_mean` is the linear average of the recorded iterate values, not the last value, not the plain mean.
- The leaf query: a recording stub is asked with the current iterate's normalised reaches (not the average's),
  for the traverser, and its answer arrives scaled by the opponent's reach.
- Search with exact leaf values reproduces the calibration at T = 64 and beats unsafe search.
- The stopping step's distribution in self-play (both settings), in `Playthrough` and in the sampled mixture.
- Exploration by exactly one player (epsilon = 1 with an iterate that always calls) and ranges that follow the
  iterate; the walk's line frequencies against the joint distribution; the board card drawn from the ranges.
- Stored examples: the root's, after the card (= an independent solve's `root_mean`, not its last iterate's
  values), before the card in both modes; the network's input layout and units; `--chance-prob`.
- The last round asks no network; the paper's chance targets ask it exactly once per agent, after a card.
- The loss on hand-made numbers, the network's layers and small initial output, the learning-rate schedule.
- The exact mixture equals a brute-force mixture of `Playthrough` policies over the game tree; the sampled
  mixture equals the mixture of its own playthroughs; a playthrough keeps one stopping step per subgame; the
  unsafe policy is the average strategy with the last round solved at its ranges.
- Kuhn end to end (exploitability = Linear CFR's), evaluation keys, an evaluation does not change the training
  trajectory, a short training lowers the value error, checkpoint round trip continues bit-identically on CPU,
  snapshots and loaded solvers do not share optimiser state, the CLI trains / resumes / calibrates.
- Added after the review, each the only guard of its property at every thread count: what `self_play` hands to the
  walk (the iterates of the stopping steps, the configured epsilon); every iterate is regret matching over the
  legal actions (exactly 0 on illegal ones, uniform where nothing is positive); the agent of the leaf query; the
  evaluation searches with the network, not with exact leaf values; the gradient norm the optimiser sees is 5;
  batches without a PBS before the card, zero minibatches, odd search lengths (refused in the constructor); the
  CLI's thread count; the curve file survives a crash while it is rewritten. The one statistical test (a short
  training lowers the value error) keeps a margin now: the ratio is 0.42-0.50 over seeds and thread counts, the
  bound 0.6.

Each wiring test was checked against one-line mutants of `rebel.py` (see "Result").

## Success criteria

Unit: mA/g = 1000 x exploitability in antes, mean over the two seats.

1. Search with exact leaf values at T = 1024 reproduces the brief's calibration: about 22 mA/g for the
   random-iterate mixture, and unsafe search clearly worse (the brief: 21.8 and 53.5).
2. A trained network: the test-time policy at T = 1024 within a factor 2 of the exact-leaf number (below about
   44 mA/g) and falling over the epochs. Reported as the mean of the last three evaluations, as the paper does
   for Liar's Dice: single evaluations of a network that is still training differ by 20-30 %.
3. Throughput: a full run within 2-3 hours on at most 8 CPU threads.

References (none is an official ReBeL-on-Leduc result):

| source | what it measures | mA/g |
|---|---|---|
| full-game tabular Linear CFR, ours | average strategy after 512 updates per player | 10.6 |
| the brief's prototype | search with exact leaf values, T = 1024: random-iterate mixture / unsafe | 21.8 / 53.5 |
| pinouche/poker_self_play (third party, single seed, unverified) | its ReBeL variant (DCFR, leaves after the card, gadget re-solving), trained net at 300 / 1 000 / 3 000 / 10 000 test-time iterations | 58.5 / 46.6 / 26.9 / 16.4 |
| the same | its exact leaf solving | 57.1 / 49.7 / 39.2 / 29.3 |
| Student of Games, Fig. 3A (read off the plot by a survey) | another algorithm (GT-CFR with learned value and policy), 100 / 1 000 simulations | about 105 / 22 |
| our SD-CFR / DREAM / Deep PDCFR+ reproductions on Leduc | exploitability of the average strategy | 44-77 |

## Compute

Measured on the shared 64-core box when it was quiet (load average 8), one process, T = 1024, torch with one
thread (the CLI's default) and with two:

| part | 1 thread | 2 threads |
|---|---|---|
| root solve, 1024 steps with 5 leaf queries each | 0.40 s | 0.42 s |
| 2048 playthroughs: about 1 900 last-round solves of 1024 steps in one batch (0.8 us per subgame and step) | 1.5 s | 1.5 s |
| 200 minibatches of 512 | 1.25 s | 0.9 s |
| one epoch: 7 600 examples = 3 800 PBSs at first (92 % of the playthroughs reach the board card), about 5 300 later | 3.2 s | 2.85 s |
| one evaluation (15 360 last-round solves of 1022 steps in one batch, three best responses) | 22 s | 22 s |

Generation alone produces 4 000 examples = 2 000 PBSs per second; the search with exact leaf values at T = 1024
(`--oracle`: 1 024 x 30 last-round solves) took 212 s on the loaded machine.

A run of 300 epochs (60 000 minibatches, 1.6 M examples) with an evaluation every 20 epochs: 16 min of training
plus 6 min of evaluations, about 22 minutes per seed on one core; the development run below took 29 minutes next
to other jobs. Three seeds fit on three cores. The official Liar's Dice length (1 000 epochs of 50 minibatches)
takes about 40 minutes plus its evaluations. `--leaf-targets solve` needs six solves per example: `--games 512`
keeps an epoch near 4 s (1 000 examples).

If 1024 steps were too expensive: `--iters 256` makes generation and evaluation four times cheaper; with exact
leaf values the search then reaches about 50 mA/g instead of 22 (the brief's calibration), which bounds what a
network trained and played that way can do. It was not needed.

The solver's cost is memory traffic, not Python: a first version with the subgame axis first (gathers along a
middle axis) needed 9 us per subgame and step; subgame-last arrays with whole-block gathers need 0.9.

## Result (2026-10-08: development measurements, seed 0; the three-seed validation is in the status line)

**Search alone, exact leaf values** (`--oracle`, exact exploitability in mA/g):

| T | random-iterate mixture | 1 024 sampled playthroughs | unsafe search | root value | the brief's prototype (mixture / unsafe) |
|---|---|---|---|---|---|
| 64 | 126.8 | - | 143.1 | -0.191 | 134.9 / 165.9 |
| 128 | 73.8 | - | 93.4 | -0.136 | 80.1 / 92.6 |
| 1024 | 22.1 | 24.8 | 54.5 | -0.099 | 21.8 / 53.5 |

Criterion 1 is met at T = 1024. The prototype's small-T numbers are not reproducible to the digit by any
independent implementation, for two reasons that were both checked: (a) it passes a range without mass down as
zeros, so the opponent of a player who never takes a line plays uniformly at random below it, where the official
code - and this solver - make the range uniform; (b) with (a) emulated and textbook regret matching the two
implementations agree to 1e-14 for the first three root steps and then drift apart by a factor 10-100 per step:
CFR-D with exact leaf solves amplifies rounding differences, and the exploitability at T = 64 moves by about 5 %
with the order of floating-point operations.

**Training, `--leaf-targets net` (the paper's targets), default settings.** Exact exploitability of the policy
played, of 1 024 sampled playthroughs, of unsafe search (mA/g); value-net RMS error on the probe set and of the
leaf values fed to the evaluated search (milli-antes):

| epoch | 1 | 35 | 70 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 220 | 240 | 260 | 280 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| policy played (exact) | 1026 | 40.5 | 52.3 | 41.0 | 49.4 | 29.8 | 30.5 | 29.6 | 36.4 | 30.3 | 24.4 | 24.8 | 25.3 | 25.1 | 26.2 |
| 1 024 sampled playthroughs | 1024 | 40.7 | 53.6 | 43.2 | 53.4 | 30.2 | 42.0 | 31.3 | 35.3 | 44.3 | 26.3 | 27.1 | 28.1 | 26.4 | 26.9 |
| unsafe search | 920 | 60.8 | 60.1 | 51.3 | 57.6 | 56.9 | 48.7 | 52.8 | 59.5 | 52.6 | 53.1 | 54.9 | 50.6 | 51.2 | 50.9 |
| value error, probe | 1032 | 244 | 221 | 204 | 205 | 193 | 196 | 180 | 171 | 174 | 165 | 169 | 160 | 161 | 161 |
| value error, fed to the search | 768 | 68 | 65 | 70 | 56 | 52 | 56 | 54 | 52 | 57 | 49 | 46 | 49 | 44 | 47 |

The mean of the last three evaluations is 25.5 mA/g: 1.15 times the exact-leaf search (22.1), against the
criterion of 2. The learning rate was halved at epochs 100 and 200; each halving lowers the plateau (about 45,
30, 25). This run was resumed three times while the code was still being polished (the solver's arithmetic was
reordered in between) and ran with two torch threads, so the final code will not reproduce it bit for bit. A
fresh 80-epoch run of the same seed with four threads (5.7 minutes) gave 994 / 113 / 56 / 41 / 29 mA/g at epochs
1 / 20 / 40 / 60 / 80, with a probe error of 991 / 333 / 264 / 237 / 220 milli-antes - another trajectory of the
same seed (see "Reproducibility"), the same course. The three validation seeds end at 27.7 +- 1.4.

The same setup with the official Liar's Dice schedule (50 minibatches per epoch, 3e-4 constant up to epoch 400):
214 / 91 / 63 / 65 / 45 / 33 / 49 / 46 / 40 / 40 / 39 mA/g at epochs 40 / 80 / ... / 240 / 260 / 280 / 300 / 320 /
340 (probe error 415 -> 217 milli-antes); stopped there.

`--leaf-targets solve` (exact targets before the card; 512 playthroughs per epoch, still with 50 minibatches and
3e-4): 34.8 / 35.0 / 26.7 / 27.6 / 42.1 / 29.3 mA/g at epochs 40 / 80 / 120 / 160 / 200 / 240, from 160 000 examples.
That is close to the default run's level (27-42 against 25) with a tenth of the examples and a fifth of the
minibatches; `net` stays the default because it is the paper's method and gets there too.

What limits the network is its fit, not the data: at epoch 280 of the official-schedule run the error on the
probe set was 0.33 antes for the values after the card, 0.20 for the bootstrapped targets before the card made
from them, and 0.23 for the network's own answers before the card. After cards the PBS makes unlikely (below
15 %) the error was 0.4, after likely ones 0.27: the self-play rarely deals them, the chance average asks for all
six.

**Sampled against exact evaluation.** The paper's protocol is a noisy upper bound of the exact number. With exact
leaf values at T = 64 (12 draws each): K = 16: 244 +- 14, K = 64: 158 +- 5, K = 256: 136.0 +- 2.4, K = 1024:
128.9 +- 1.0 mA/g against 126.8 exact, single K = 1024 draws scatter by 3.5. At T = 1024 a root iterate is drawn
only twice on average: in the training table above the K = 1024 number lies up to 14 mA/g above the exact one
and twice 1-2 below it; in other runs it fell up to 6 below. It is an upper bound in expectation only.

**Mutation check.** One-line mutants of `rebel.py`, each run against `tests/test_rebel.py` in its own copy of the
package. First pass: 120 mutants, 115 killed. Of the five survivors one was equivalent (a reordered sum), one
showed dead state (the torch generator was saved and restored although nothing draws from it after the network's
initialisation: removed), three were real gaps (all subgames of a sampled playthrough sharing one stopping step,
the regret-floor setting not reaching the solvers, the root value's normalisation) and got assertions. Final
pass on the final code: 130 mutants, 129 killed; the survivor (the unsafe comparison policy's last round solved
for half the steps) got its own test and is killed too.

The review then showed that this pass had run with one torch thread only: two mutants (no exploration in
self-play, regret matching over illegal actions) were killed by nothing but the statistical value-error test,
by a margin inside its noise, and survived with 2 or 4 threads; a third (the evaluation searching with exact leaf
values) had not been in the list. Each has its direct test now and is killed by it with 1, 2 and 4 threads, as
are the mutants of the code added in the fix pass (16 mutants run again).

**Not verified:** `--device cuda:0` (the session was CPU-only by instruction); the hold'em stages of the brief.
