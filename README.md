# headsup-poker

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
python -m pytest tests -q                  # 38 tests, ~12 s
```

Device selection is automatic (`mps` → `cuda` → `cpu`); override with `--device` or
`HEADSUP_DEVICE=cpu`. Torch ≥ 2.4 (the C++ build was tested with Apple clang; setup.py picks
it automatically on macOS).

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
first on later streets. Reward = chips won per hand (1 chip = 500 mbb at blinds 1/2).
Observations are `float32[31]`: hand (2 × [rank+1, suit+1, card+1]), board (5 × same,
0-padded), stage, position, 8 normalised bet/stack features. The encoding is unchanged from
the original project, so old checkpoints still load.

## DeepCFR training

```bash
python -m headsup.deepcfr.train --out runs/deepcfr --checkpoint-every 10          # 300 it., paper-like
python -m headsup.deepcfr.train --iterations 100 --value-steps 2000 --out runs/quick   # ~45 min on an M2 Max
python -m headsup.deepcfr.train --resume runs/deepcfr/checkpoint.pt --iterations 300 --out runs/deepcfr
python -m headsup.deepcfr.train --policy-only runs/deepcfr/checkpoint.pt --out runs/deepcfr   # re-fit the policy net
tensorboard --logdir runs
```

Per iteration and seat: 10 000 external-sampling traversals (C++, all cores) → advantage
samples into a reservoir memory on the GPU → the seat's advantage network is re-fitted from
scratch (4000 steps × 16 384). At the end (or on `Ctrl-C`):

- **DeepCFR** (`--algo deepcfr` / `both`): the average-strategy network is fitted on the
  strategy memory → `<out>/policy.pth` (player spec `cfr:<out>/policy.pth`);
- **SD-CFR** (`--algo sdcfr` / `both`): all iterates are written to `<out>/iterates.pt`
  (player spec `sdcfr:<out>/iterates.pt`, or `…@exact` for exact per-infoset averaging
  instead of per-hand trajectory sampling — same distribution, see `headsup/sdcfr.py`).

Both are evaluated against simple opponents and, with `both`, head-to-head (`eval.json`).
On an M2 Max an iteration takes ≈ 40–55 s (traversals grow as the bots stop folding).

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

`rl_games_env.py` registers `headsup_poker` with rl_games using our vectorised env (all
tables in one process, opponent inference batched / native in C++). `headsup.exploit` trains
several PPO best responses (different seeds, in parallel), exports them to ONNX and
evaluates each against the policy with argmax and sampled play over many hands; the largest
exploiter reward is the number to quote (a lower bound on exploitability, ± SE):

```bash
python -m headsup.exploit --policy cfr --seeds 3 --epochs 1000 --hands 400000 --out runs/exploit_cfr
python -m headsup.exploit --policy sdcfr:runs/deepcfr/iterates.pt --seeds 3
python -m headsup.exploit --policy cfr --onnx models/rl_games_exploiter.onnx     # evaluate existing exploiters
```

The lower-level pieces are still available: `exploitability.py -t/-p` (one exploiter,
rl_games CLI semantics) and `rl_games_onnx.py` (export). `--device mps` works, but the
exploiter's MLP is tiny and CPU is faster.

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
headsup/engine.py        game rules, observation encoding, cheap clone()
headsup/env.py           PokerVecEnv / SingleAgentEnv (Python), NativeVecEnv (C++), rl_games IVecEnv API
headsup/players.py       batched players: random/call/allin/raise, torch policy, regret-matching iterate, numpy, onnx
headsup/model.py         DeepCFR network (torch); numpy_model.py = numpy mirror for CPU workers
headsup/cpp/             C++ kernels (pybind11): engine, evaluator, MLP, MCCFR traversal, VecEnv
headsup/deepcfr/         memory.py (reservoir), traverse.py, train.py, evaluate.py
headsup/sdcfr.py         Single Deep CFR: iterate bank + average-strategy player (exact / trajectory sampling)
headsup/compare.py       head-to-head comparison CLI;  headsup/exploit.py: multi-seed PPO exploitability CLI
headsup/web/             browser table: server.py (http.server), session.py (game logic), static/ (HTML/CSS/JS)
rl_games_env.py          rl_games registration;  exploitability.py / rl_games_onnx.py: exploiter tools
models/                  deepcfr_policy.pth, rl_games_exploiter.onnx
tests/                   pytest suite (rules, C++ ⇔ Python equivalence, envs, memories, web API)
```

## How close is this to the papers?

**Matches DeepCFR** (Brown et al. 2019): external-sampling traversals (all actions at the
traverser's infosets, sampled opponent/chance), advantage samples `v(a) − Σσv` weighted by
the iteration (linear CFR), advantage nets re-initialised and trained from scratch every
iteration (Adam 1e-3, grad-norm clip 1, 4000 minibatch steps), reservoir memories, the
average-strategy net fitted with `t`-weighted MSE, and the paper's card-embedding network.
**Matches SD-CFR** (Steinberger 2019): all iterates kept, exact reach-weighted linear
average at play time and the trajectory-sampling variant.

**Deviations** (inherited from the original project or chosen for this game): the bet
features are 8 aggregated numbers instead of the paper's per-bet round history (a mild
imperfect-recall abstraction — the raise count is still recoverable from the street bets
here); regret matching falls back to uniform when no advantage is positive; memories are
10 M instead of 40 M samples; batch 16 384 instead of 10 000; the game itself is a small
action abstraction of no-limit hold'em (fold / call / min-raise / all-in, 3rd raise → all-in)
rather than HULH/FHP; exploitability is a PPO best-response lower bound, not exact.

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
- Tools: [treys](https://github.com/ihendley/treys) (hand evaluator),
  [rl_games](https://github.com/Denys88/rl_games) (PPO), [pybind11](https://github.com/pybind/pybind11)
