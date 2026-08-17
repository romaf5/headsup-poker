# PLAN.md — next steps

Status: engine, C++ kernels, DeepCFR + SD-CFR trainer, evaluation/exploitability tools and the
browser UI are done and tested (see CLAUDE.md). The last completed model was trained on the old
rules (free fold allowed); the no-free-fold rule is now in place and everything should be retrained.

### Progress log (2026-08-16, 64-core / 2x RTX 3090 box)

- Step 0: `runs/base` (aggregated features, current net, 300 it. x 40k traversals, 20 M memories)
  trained; evaluation (compare 1.5 M hands, exploit 3 seeds, LBR) → README results table.
- Step 1: DONE in code — `--features aggregated|history|both`, `--net current|paper`, `--cards embed|onehot`,
  `--dim`, `--rm-fallback uniform|argmax`, `--preset paper` (all differential-tested torch/numpy/C++,
  paper facts verified from arXiv 1811.00164 / 1901.07621, see README "How close..."). Ablation arms
  under `runs/abl_*` ({aggregated, history} x {current, paper} + history/paper/argmax), same budget as
  the base run; results → README.
- Step 2: DONE — `headsup/lbr.py` (+ C++ `equity_vs_all`), duplicate hands, `--model-iterates` thinning
  for SD-CFR banks, trainer `--lbr-every` / final `lbr_final/*`; `@t<N>` / `iterate:` specs for curves.
- Step 3: DONE in code — `headsup/game.py` GameConfig (`--bet-sizes 0.5,1,2 --raise-cap --mask-redundant`),
  engines / traversal / players / envs / SD-CFR / LBR / web UI / rl env generic in `num_actions`
  (default 4-action game bit-identical). Not yet trained (see Step 3 below: retrain, more traversals).
- Step 4: `--regret-power` (t^alpha sample weights, DCFR alpha = 1.5) and `--strategy-power` (gamma = 2)
  options added; real-time search DONE in code (headsup/search.py: unsafe depth-limited subgame solving,
  sampled LCFR pre-river + vector-form DCFR/CFR+/PCFR+ river solver, verified with exact best responses);
  its evaluation vs the blueprint is pending; DREAM / ESCHER not started.

## Step 0 — retrain on the fixed rules (first thing on the new machine)

```bash
python -m headsup.deepcfr.train --algo both --iterations 300 --traversals 40000 --checkpoint-every 10 \
    --adv-capacity 20000000 --strat-capacity 20000000 --eval-hands 400000 --out runs/base
python -m headsup.exploit --policy cfr:runs/base/policy.pth --seeds 3 --hands 400000
python -m headsup.exploit --policy sdcfr:runs/base/iterates.pt --seeds 3 --hands 400000
python -m headsup.compare cfr:runs/base/policy.pth sdcfr:runs/base/iterates.pt sdcfr:runs/base/iterates.pt@g2 cfr --hands 1500000 --bots
```
Ship the better of {policy net, SD-CFR bank}, update `models/`, `models/README.json`, README results.
(1.5 M hands are needed to resolve paper-sized DeepCFR-vs-SD-CFR differences of 10–30 mbb/g.)

## Step 1 — paper-faithful features and architecture

Goal: reproduce the observation features and network of Brown et al. (2019) / Steinberger (2019)
exactly, as a switchable variant, and measure against the current design in the same run.

1. Bet-history features (paper): for every betting round and each of the first N actions in it,
   `[bet size / pot at that time, 1 if an action occurred]` (DeepCFR uses N per round; verify N and
   the pot normalisation in the paper's Section 5 / appendix), plus stack, pot, position. Keep the
   current 8 aggregated features as an alternative (`--features aggregated|history`).
   - Engine: record the per-street action list (who, size) — Python + C++ + tests; observation
     length becomes a parameter (OBS_DIM); C++ VecEnv / traversal / Model input widths follow.
   - Card representation: DeepCFR = rank/suit/card embeddings summed per group (current);
     SD-CFR paper = concatenated one-hot cards. Offer both (`--cards embed|onehot`).
2. Network sizes exactly as the DeepCFR paper's Figure 1 (verify layer widths and the ~98,948-parameter
   count from the paper; ours is 67,844 with 64-wide layers). Make width/embedding dim CLI options.
3. Regret matching fallback: paper picks the highest-advantage action when no advantage is positive
   (verify wording); add `--rm-fallback uniform|argmax`.
4. Paper hyperparameters as a preset: K = 10 000 traversals, batch 10 000, 4000 updates, memories
   40 M, strategy net 20 000 updates x batch 20 480 (SD-CFR paper, 5-FHP), 300–450 iterations.
5. Ablation run with `--algo both`: {current features, paper features} x {current net, paper net},
   compare with `headsup.compare` (1.5 M hands) and `headsup.exploit` (3 seeds each). Record in README.

## Step 2 — Local Best Response (LBR) evaluator

Lisý & Bowling (2017): a best-response-style exploiter that knows the opponent's policy, maintains
its hand range from the observed actions, and picks the action maximising expected value under a
one-street lookahead (call/fold value = equity of the range x pot odds; raise = fold-equity + ...).
Deterministic, cheap, much stronger than PPO at finding leaks; the standard lower bound for big games.
- `headsup/lbr.py`: range tracking (Bayes update with the policy's probs over all 1326 hole-card
  combos, C++ vmapped forward), EV of check/call and of raise/all-in with the current street's
  cards; CLI `python -m headsup.lbr --policy spec --hands N` reporting mbb/g ± SE.
- Report LBR next to the PPO bound in README; use it as the training-time exploitability signal
  (`--lbr-every N` in the trainer) instead of the simple-bot evals.

## Step 3 — a realistic action abstraction

fold / call / min-raise / all-in is effectively a limit game with a shove. Add bet sizes
(e.g. 0.5 pot, 1 pot, 2 pot, all-in) with a max-raises-per-street cap: engine (Python + C++),
`NUM_ACTIONS` as a parameter, action heads, C++ traversal branching, UI buttons ("Bet 12"),
players' masks (illegal sizes), rl_games env. Retrain; expect much bigger trees — raise
`--traversals`, and consider batching network inference across roots on the GPU.

## Step 4 — algorithmic upgrades (in order of effort)

- DCFR discounting for regrets (alpha = 1.5, beta = 0) in addition to the strategy weighting gamma = 2.
- Bigger memories (40 M+), more iterations (500+), several seeds; keep `--algo both`.
- DREAM (Steinberger, Lerer, Brown 2020): outcome sampling + learned baseline for variance
  reduction — attacks the sampled-regret noise floor (`advantage/*/mse_unweighted`).
- ESCHER (McAleer et al. 2022): DREAM without importance weights; unbiased/low variance.
- Real-time search: depth-limited subgame re-solving on the current street with the blueprint
  (DeepCFR/SD-CFR) values at the leaves — the biggest strength gain for actual play. Must follow a
  paper (Brown & Sandholm 2018 "Depth-Limited Solving for Imperfect-Information Games" / Libratus
  nested subgame solving; DeepStack continual re-solving) as a proper CFR tree search over the
  subgame, verified against tabular CFR on small subgames — not an ad-hoc river solver.

## Step 5 — engineering nice-to-haves

- GPU-batched traversals (many roots at once) for the bigger action space.
- rl_games exploiter on CUDA (`--device cuda:0`), longer schedules, PPO with action masks.
- Web UI: show LBR / exploitability numbers, session export, keyboard-only play polish.
