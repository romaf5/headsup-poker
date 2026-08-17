# headsup-poker

[![tests](https://github.com/romaf5/headsup-poker/actions/workflows/tests.yml/badge.svg)](https://github.com/romaf5/headsup-poker/actions/workflows/tests.yml)

Heads-up no-limit-style Texas Hold'em: a fast game engine, **DeepCFR** training on Apple
Silicon (MPS) / CUDA / CPU, exploitability evaluation with a PPO best-response agent, and a
browser table to play against the bots — with a DeepCFR advisor at your side.

![Browser table](imgs/web-ui.png)

## Highlights

- **One engine** for everything (`headsup/engine.py`): rules, chips, showdown, cheap `clone()`
  for tree search. Blinds 1/2, stacks 100 (configurable), actions fold / check-call /
  min-raise / all-in; the 3rd raise in a row becomes an all-in so the game tree stays finite.
- **C++ kernels** (`headsup/cpp/`, pybind11): engine, treys-compatible hand evaluator, the
  DeepCFR network forward pass, external-sampling MCCFR traversals (GIL released → all cores)
  and an envpool-style vectorised env. ~50× faster traversals than Python; every kernel is
  differential-tested against its Python twin.
- **DeepCFR trainer** (`headsup/deepcfr/`): reservoir memories on the GPU, `torch.compile`,
  checkpoint/resume, Ctrl-C-safe, TensorBoard with signals that actually mean something.
- **Single Deep CFR** (Steinberger 2019, `headsup/sdcfr.py`): the same run also keeps every
  iteration's advantage net and plays the exact linear-CFR average strategy from them — no
  strategy memory / policy net approximation. `--algo deepcfr|sdcfr|both` picks the variant;
  `both` gives an apples-to-apples comparison from one training run.
- **Comparison & exploitability tools**: `headsup.compare` (round-robin head-to-head with
  standard errors) and `headsup.exploit` (several PPO best responses in parallel, argmax and
  sampled play, hundreds of thousands of hands, max = exploitability lower bound).
- **Browser UI** (`headsup/web/`, no JS build step, dependency-free HTTP server): play vs.
  the DeepCFR bot or the exploiter, see the bot's action distribution, get advice + Monte-Carlo
  equity from the DeepCFR policy, autoplay, hand history, session sparkline.

## Setup

```bash
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements.txt            # engine, training, evaluation, web UI
pip install -r requirements-rl.txt         # + rl_games exploitability, onnx (optional)
python setup.py build_ext --inplace        # C++ kernels (needs a C++17 compiler; optional but recommended)
python -m pytest tests -q                  # 60 tests, ~45 s
```

Device selection is automatic (`mps` → `cuda` → `cpu`); override with `--device` or
`HEADSUP_DEVICE=cpu`. Torch ≥ 2.4 (the C++ build was tested with Apple clang; setup.py picks
it automatically on macOS). On CUDA the network fits are captured into CUDA graphs
(`torch.compile(mode="reduce-overhead")`: the 64-wide MLP is launch-bound, ~1.7× faster).

## Play in the browser

```bash
python -m headsup.web                                # http://127.0.0.1:8000/  vs the DeepCFR policy
python -m headsup.web --opponent onnx                # vs the PPO exploiter
python -m headsup.web --opponent call --advisor ''   # calling station, no advisor
python -m headsup.web --opponent cfr:runs/deepcfr_full/policy.pth --port 8080
```

Keys: `F` fold, `C` check/call, `R` bet/raise, `A` all-in, `Enter` next hand, `S` peek at
the bot's cards, `D` toggle the advisor, `P` autoplay. The ⚙ Table dialog switches
opponent / advisor / stacks / blinds / raise cap / seed without restarting.

## The game & observations

Seat 0 is dealer/small blind and acts first pre-flop; seat 1 posts the big blind and acts
first on later streets. Actions: fold, check/call, one raise per configured **bet size**, all-in.
The default game has a single size — the min-raise (call + one big blind) — and the third
raise in a row becomes an all-in (finite tree). `--bet-sizes 0.5,1,2` (trainer) turns it into
a no-limit-style abstraction with half-pot / pot / 2×-pot raises (always at least a min-raise,
capped by the stack; `raise_cap` consecutive raises → all-in) where raises that merely
duplicate another action (reach the stack, coincide with a smaller size, or are the capped
raise) are hidden from the tree (`mask_redundant`, on by default for custom sizes). The action
tree (`bet_sizes`, `raise_cap`, `mask_redundant`, see `headsup/game.py`) is stored in every
model's config, so envs, the UI and the tools build the right game for a given bot. Folding only
exists when facing a bet — with nothing to call it would be a dominated check, and leaving it
in the tree let early CFR iterations put lasting weight on it (the average strategy folded
~17 % of flops for free before this rule). Reward = chips won per hand (1 chip = 500 mbb at
blinds 1/2).
Observations are `float32[80]`: hand (2 × [rank+1, suit+1, card+1], sorted), board (5 × same,
flop sorted, 0-padded), stage, position, 8 normalised bet/stack features (the original 31
features), then the DeepCFR paper's bet history: for each of the 4 betting rounds and each of
its first 6 actions `[chips put in / pot before the action, 1 = an action occurred]`, and the
number of consecutive raises on the current street (index 79, needed to derive legal actions
from an observation). Every network reads the prefix it was trained on (`features=aggregated`
→ the first 31, `history`/`both` → 79), so envs and players never need to know which variant
they are talking to.

## DeepCFR training

```bash
python -m headsup.deepcfr.train --out runs/deepcfr --checkpoint-every 10          # 300 it., paper-like
python -m headsup.deepcfr.train --iterations 100 --value-steps 2000 --out runs/quick   # ~45 min on an M2 Max
python -m headsup.deepcfr.train --resume runs/deepcfr/checkpoint.pt --iterations 300 --out runs/deepcfr
python -m headsup.deepcfr.train --policy-only runs/deepcfr/checkpoint.pt --out runs/deepcfr   # re-fit the policy net
tensorboard --logdir runs
```

Per iteration and seat: 10 000 external-sampling traversals (C++, all cores) → advantage
samples into a reservoir memory on the GPU (compact: uint8 card/stage features + float32 bet
features, 75 B per 31-feature sample) → the seat's advantage network is re-fitted from
scratch (4000 steps × 16 384). At the end (or on `Ctrl-C`):

- **DeepCFR** (`--algo deepcfr` / `both`): the average-strategy network is fitted on the
  strategy memory → `<out>/policy.pth` (player spec `cfr:<out>/policy.pth`);
- **SD-CFR** (`--algo sdcfr` / `both`): all iterates are written to `<out>/iterates.pt`
  (player spec `sdcfr:<out>/iterates.pt`, or `…@exact` for exact per-infoset averaging
  instead of per-hand trajectory sampling — same distribution, see `headsup/sdcfr.py`;
  `…@g2` quadratic iterate weights, `…@t100` the average after 100 iterations, `…@k64` a bank
  thinned to 64 representative iterates; `iterate:<out>/iterates.pt@t100` plays iteration 100's
  current strategy).

Both are evaluated against simple opponents and, with `both`, head-to-head (`eval.json`).
On an M2 Max an iteration takes ≈ 40–55 s (traversals grow as the bots stop folding), on a
64-core box with an RTX 3090 ≈ 25–30 s (the fit dominates; traversals take ~1–3 s).

Network / algorithm variants (all differential-tested torch ⇔ numpy ⇔ C++; the config is
stored in every artefact — `policy.pth`, `iterates.pt`, checkpoints — so players pick it up):

| flag | choices | meaning |
|---|---|---|
| `--features` | `aggregated` (default) / `history` / `both` | bet branch input: the 8 pot-normalised totals, or the DeepCFR paper's per-street bet history (+ stack/pot, pot), or both |
| `--net` | `current` (default) / `paper` | 3 branches (cards, stage+position embeddings, bets) → 3·dim trunk, or Brown et al. Fig. 1: cards + bets only, one card embedding per card group, position appended to the bet features |
| `--cards` | `embed` (default) / `onehot` | rank+suit+card embeddings summed per group (DeepCFR) or concatenated one-hot cards into the first card layer (SD-CFR paper) |
| `--dim` | int (64) | width of every hidden layer / embedding (64: 67 844 params current, 67 524 paper) |
| `--rm-fallback` | `uniform` (default) / `argmax` | regret matching when no advantage is positive: uniform over the allowed actions, or the highest advantage with probability 1 (DeepCFR paper, Fig. 4) |
| `--preset` | `default` / `paper` | `paper` = 10 000 traversals, batch 10 000, 40 M memories, policy net 20 000 updates × 20 480 (explicit flags win) |
| `--bet-sizes`, `--raise-cap`, `--mask-redundant` | `min` (default) / pot fractions e.g. `0.5,1,2` | the action tree (see above); stored in the model config |
| `--regret-power`, `--strategy-power` | float (1) | sample weights `t^power` in the advantage / policy fits (1 = linear CFR; DCFR-style 1.5 / 2) |

What to watch in TensorBoard:

| tag | meaning |
|---|---|
| `eval_current_strategy/*` | current CFR iterate (regret matching on the advantage nets) vs random / call / all-in, chips/hand, every 5 it. |
| `eval_avg_strategy/*` | quick fit of the DeepCFR *average* strategy — the thing CFR converges — every 25 it. |
| `eval_sdcfr/*` | the SD-CFR average strategy (no fitting needed) vs the same bots, every 25 it. |
| `advantage/seat*/final_loss` | the training loss; it is weighted by the iteration `t` (linear CFR) so it grows ~linearly by design |
| `advantage/seat*/mse_unweighted`, `target_rms` | unweighted fit error vs the scale of the sampled regrets (single-sample targets are very noisy, so the ratio stays high) |
| `samples/nodes_per_traversal` | game-tree size per traversal — grows as the bots stop folding |

Evaluate any policy against the simple bots (chips/hand, batched, native env), or compare
policies head-to-head (round-robin, alternating seats, ± standard error):

```bash
python -m headsup.deepcfr.evaluate --policy cfr --hands 200000
python -m headsup.compare cfr sdcfr:runs/deepcfr/iterates.pt cfr:models/deepcfr_policy_v1.pth --hands 400000 --bots
```

## Exploitability (best-response lower bound with rl_games)

`headsup/rl/env.py` registers `headsup_poker` with rl_games using our vectorised env (all
tables in one process, opponent inference batched / native in C++). `headsup.exploit` trains
several PPO best responses (different seeds, in parallel), exports them to ONNX and
evaluates each against the policy with argmax and sampled play over many hands; the largest
exploiter reward is the number to quote (a lower bound on exploitability, ± SE):

```bash
python -m headsup.exploit --policy cfr --seeds 3 --epochs 1000 --hands 400000 --out runs/exploit_cfr
python -m headsup.exploit --policy sdcfr:runs/deepcfr/iterates.pt --seeds 3
python -m headsup.exploit --policy cfr --onnx models/rl_games_exploiter.onnx     # evaluate existing exploiters
```

The lower-level pieces are still available: `python -m headsup.rl.exploitability -t/-p` (one
exploiter, rl_games CLI semantics) and `python -m headsup.rl.onnx` (export). `--device mps` works, but the
exploiter's MLP is tiny and CPU is faster.

## Local Best Response (LBR)

`headsup.lbr` implements Lisý & Bowling's *Local Best Response*: an agent that knows the
opponent's strategy, keeps a Bayesian range over the opponent's 1326 hole-card combinations
(the strategy is queried on the opponent's observation with every possible hand substituted),
computes its equity against that range with a C++ kernel (all runouts on the turn/river,
Monte-Carlo before) and picks fold / check-call / min-raise / all-in by a one-street lookahead
(bet actions use the opponent's actual fold probability per combo, otherwise the hand is
checked down). Deterministic, no training, and much stronger than a PPO best response at
finding leaks; the result is a lower bound on exploitability like the PPO one. Hands are played
in duplicate (same deal, seats swapped) so card luck mostly cancels.

```bash
python -m headsup.lbr --policy cfr:runs/x/policy.pth --hands 50000                    # 50k duplicate pairs, ~10 min
python -m headsup.lbr --policy sdcfr:runs/x/iterates.pt --hands 20000 --model-iterates 32
python -m headsup.deepcfr.train ... --lbr-every 25 --lbr-hands 1000                   # LBR curves in TensorBoard
```

For SD-CFR banks the queried model can be thinned to `--model-iterates K` representative
iterates (equal-weight bins) — the opponent still plays the full bank, LBR just becomes a bit
weaker, so the bound stays valid; T = 300 iterates cost ~4 ms per iterate per 64-table query.
The trainer logs `lbr/current_strategy`, `lbr/sdcfr` (and, at the end, `lbr_final/*` incl. the
policy net) — LBR's chips/hand, lower is better.

## Real-time search (depth-limited subgame solving)

`headsup/search.py` + `headsup_cpp.SubgameSolver` / `RiverSolver` implement search at play time
the way Libratus / Modicum / Pluribus do it: at every decision the game from the current public
state is re-solved with CFR ("unsafe subgame solving", Brown & Sandholm 2017), the root dealing
both players' hands from their ranges — the opponent's reach under the blueprint, the hero's under
the strategies it actually played (nested re-solving) — and depth-limited to the end of the
current street (Brown, Sandholm & Amos 2018): street-end leaves are rolled out with a
continuation strategy (the blueprint policy net, the last SD-CFR iterate or a thinned bank of
iterates, sampled per rollout). Pre-river subgames are solved by external-sampling MCCFR with
tabular regrets and linear averaging, dealing the hero's real hand on half of its own
traversals; the river is solved exactly by full-width vector-form CFR over all 1326 hands
(DCFR by default; LCFR / CFR+ / PCFR+ selectable). `headsup.search.exploitability` computes the
exact best response of both players in a river subgame, which is how the solvers are verified
(tests): on a checked-down river with pot 4 / stacks 98 the exploitability of the solved profile
is 2.7 chips per hand pair for uniform play, 0.27 after 800k sampled LCFR iterations (2.8 s) and
0.003 after 200 vector-form DCFR iterations (0.45 s); with sampled regrets LCFR beat DCFR / CFR+ /
PCFR+ (0.27 vs 0.73 / 0.78 / 0.76), with full-width updates DCFR / CFR+ / PCFR+ beat LCFR
(0.0033 / 0.0080 / 0.0068 vs 0.0123 at 200 iterations).

```bash
python -m headsup.compare search:runs/x/policy.pth@it20000 cfr:runs/x/policy.pth --hands 2000
python -m headsup.web --opponent search:runs/x/iterates.pt@contbank@thin8   # search on top of SD-CFR
```

Later decisions on the same street are warm-started from the previous solve's regrets (scaled
to a few thousand equivalent iterations, Brown & Sandholm 2016 style): on the checked-down river
example, re-solving after check / bet reaches 0.98 instead of 4.3 chips of exploitability at 5k
iterations. Player spec:
`search:<blueprint>[@it<N>][@rit<N>][@rv<lcfr|dcfr|cfr+|pcfr+>][@focus<f>][@cont<policy|iterate|bank>][@thin<K>][@warm<N>]`.
The public state (bet history) is rebuilt from the observation (`headsup/public.py`), so the
search player works everywhere a player does (envs, UI, LBR, PPO exploiters). A decision costs
~1–2 s pre-river at 20k iterations (tables are solved in parallel threads) and ~0.5 s on the river.

**About the old "500 mbb/g".** The original `poker_env.py` (used for the exploiter) had no
raise cap while the DeepCFR training env converted the 3rd consecutive raise into an all-in,
so the exploiter partly learned to raise repeatedly into lines the bot had never seen. With
uncapped raising the shipped exploiter wins ≈ +1.05 chips/hand (≈ 520 mbb/g) against the
shipped policy, but *loses* ≈ 0.67 chips/hand in the game the bot was trained on. Both sides
now share one engine; `env_config.raise_cap` selects the game (`1000000` = uncapped).

## Results

Chips/hand (1 chip = 500 mbb), measured with this code (`headsup.deepcfr.evaluate`, 200k–400k
hands, ± ≈ 0.05). "Exploiter" = the larger reward of two PPO best responses (1000 epochs each,
argmax play), a lower bound on exploitability.

| policy | vs random | vs call | vs all-in | exploiter |
|---|---|---|---|---|
| `models/deepcfr_policy.pth` — 300 it., 40k traversals/it., 3.6 h on an M2 Max | **+3.70** | **+6.26** | **+3.00** | **+0.31** (≈ 150 mbb/g) |
| `models/deepcfr_policy_v1.pth` — original 300-it. model | +3.61 | +4.72 | +2.27 | +0.37 (≈ 185 mbb/g) |

Head-to-head the two are tied (−0.06 ± 0.05 chips/hand for the new one), as expected for
two near-equilibrium strategies. `models/rl_games_exploiter.onnx` is the exploiter trained
against the shipped policy; `models/README.json` records the training settings.

## Layout

```
headsup/engine.py        game rules, observation encoding, legal-action masks, cheap clone()
headsup/game.py          GameConfig: bet sizes / raise cap / redundant-raise masking = the action tree
headsup/env.py           PokerVecEnv / SingleAgentEnv (Python), NativeVecEnv (C++), rl_games IVecEnv API
headsup/players.py       batched players: random/call/allin/raise, torch policy, regret-matching iterate, numpy, onnx
headsup/model.py         DeepCFR network (torch); numpy_model.py = numpy mirror for CPU workers
headsup/cpp/             C++ kernels (pybind11): engine, evaluator, MLP, MCCFR traversal, VecEnv
headsup/deepcfr/         memory.py (reservoir), traverse.py, train.py, evaluate.py
headsup/sdcfr.py         Single Deep CFR: iterate bank + average-strategy player (exact / trajectory sampling)
headsup/compare.py       head-to-head comparison CLI;  headsup/exploit.py: multi-seed PPO exploitability CLI
headsup/lbr.py           Local Best Response (range tracking, equity vs range, one-street lookahead) CLI
headsup/search.py        real-time search player (subgame re-solving; C++ SubgameSolver / RiverSolver), exploitability check
headsup/public.py        rebuild the public state (engine replay) from an observation
headsup/web/             browser table: server.py (http.server), session.py (game logic), static/ (HTML/CSS/JS)
headsup/rl/              rl_games registration (env.py), exploiter CLI (exploitability.py), ONNX export (onnx.py)
models/                  deepcfr_policy.pth, rl_games_exploiter.onnx
tests/                   pytest suite (rules, C++ ⇔ Python ⇔ numpy equivalence for every network variant, envs, memories, SD-CFR, web API)
```

## How close is this to the papers?

**Matches DeepCFR** (Brown et al. 2019): external-sampling traversals (all actions at the
traverser's infosets, sampled opponent/chance), advantage samples `v(a) − Σσv` weighted by
the iteration (linear CFR), advantage nets re-initialised and trained from scratch every
iteration (Adam 1e-3, grad-norm clip 1, 4000 minibatch steps), reservoir memories, the
average-strategy net fitted with `t`-weighted MSE, and the paper's card-embedding network.
**Matches SD-CFR** (Steinberger 2019): all iterates kept, exact reach-weighted linear
average at play time and the trajectory-sampling variant.

**Switchable, verified against the papers** (`--features`, `--net`, `--cards`,
`--rm-fallback`, `--preset paper`, see the table above): the bet history exactly as described
in DeepCFR §5.1 ("in each of the N_rounds rounds of betting there can be at most 6 sequential
actions … each betting position is encoded by a binary value specifying whether a bet has
occurred, and a float value specifying the bet size"; we normalise the size by the pot at
that time), the Appendix C network (three card layers, two bet layers, three trunk layers
with skip connections and last-layer normalisation; one `CardEmbedding` per card group; the
paper's stated 98 948 parameters cannot be reproduced from its own Appendix C code at any
integer width, so we keep 64), the paper's regret-matching fallback ("we choose the action
with highest counterfactual regret with probability 1", ~50 % lower exploitability than uniform
in their Fig. 4), and its hyperparameters (K = 10 000 traversals, batch 10 000, 4 000 SGD
steps, 40 M memories; SD-CFR: average-strategy net 20 000 updates × 20 480, cards as
concatenated one-hot vectors, 300 000 traversals per iteration for 5-FHP).

**Deviations** (design choices for this game): the default bet features are 8 aggregated
numbers (a mild imperfect-recall abstraction; the paper-faithful history is one flag away);
regret matching falls back to uniform by default; SD-CFR's iterate weights are `t^γ` (γ = 1
linear, `@g2` for DCFR-style quadratic weighting); memories default to 10 M (`--preset paper`
= 40 M) and the batch to 16 384; the game itself is a small action abstraction of no-limit
hold'em (fold / call / min-raise / all-in, 3rd raise → all-in) rather than HULH/FHP;
exploitability is a PPO best-response lower bound, not exact.

## References

- N. Brown, A. Lerer, S. Gross, T. Sandholm. *Deep Counterfactual Regret Minimization.*
  ICML 2019. [arXiv:1811.00164](https://arxiv.org/abs/1811.00164)
- E. Steinberger. *Single Deep Counterfactual Regret Minimization.* 2019.
  [arXiv:1901.07621](https://arxiv.org/abs/1901.07621)
- E. Steinberger, A. Lerer, N. Brown. *DREAM: Deep Regret minimization with Advantage
  baselines and Model-free learning.* 2020. [arXiv:2006.10410](https://arxiv.org/abs/2006.10410)
  (learned baselines for the sampled-regret variance — the natural next step here)
- M. Zinkevich, M. Johanson, M. Bowling, C. Piccione. *Regret Minimization in Games with
  Incomplete Information.* NeurIPS 2007 (CFR)
- M. Lanctot, K. Waugh, M. Zinkevich, M. Bowling. *Monte Carlo Sampling for Regret
  Minimization in Extensive Games.* NeurIPS 2009 (external sampling MCCFR)
- V. Lisý, M. Bowling. *Equilibrium Approximation Quality of Current No-Limit Poker Bots.*
  AAAI-17 Workshop on Computer Poker and Imperfect Information Games, 2017.
  [arXiv:1612.07547](https://arxiv.org/abs/1612.07547) (Local Best Response)
- N. Brown, T. Sandholm. *Safe and Nested Subgame Solving for Imperfect-Information Games.*
  NeurIPS 2017. [arXiv:1705.02955](https://arxiv.org/abs/1705.02955) (subgame solving, nesting)
- N. Brown, T. Sandholm, B. Amos. *Depth-Limited Solving for Imperfect-Information Games.*
  NeurIPS 2018. [arXiv:1805.08195](https://arxiv.org/abs/1805.08195) (depth limit + continuation strategies)
- N. Brown, T. Sandholm. *Solving Imperfect-Information Games via Discounted Regret Minimization.*
  AAAI 2019. [arXiv:1809.04040](https://arxiv.org/abs/1809.04040) (DCFR, LCFR)
- G. Farina, C. Kroer, T. Sandholm. *Faster Game Solving via Predictive Blackwell Approachability:
  Connecting Regret Matching and Mirror Descent.* AAAI 2021. [arXiv:2007.14358](https://arxiv.org/abs/2007.14358) (PCFR+)
- Tools: [treys](https://github.com/ihendley/treys) (hand evaluator),
  [rl_games](https://github.com/Denys88/rl_games) (PPO), [pybind11](https://github.com/pybind/pybind11)
