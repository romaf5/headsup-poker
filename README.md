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
- **Exploitability**: rl_games PPO exploiter against a frozen policy through our own
  vectorised env (no Ray), ONNX export of the exploiter.
- **Browser UI** (`headsup/web/`, no JS build step, dependency-free HTTP server): play vs.
  the DeepCFR bot or the exploiter, see the bot's action distribution, get advice + Monte-Carlo
  equity from the DeepCFR policy, autoplay, hand history, session sparkline.

## Setup

```bash
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements.txt            # engine, training, evaluation, web UI
pip install -r requirements-rl.txt         # + rl_games exploitability, onnx (optional)
python setup.py build_ext --inplace        # C++ kernels (needs a C++17 compiler; optional but recommended)
python -m pytest tests -q                  # 32 tests, ~7 s
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
scratch (4000 steps × 16 384). At the end (or on `Ctrl-C`) the average-strategy network is
fitted on the strategy memory, saved to `<out>/policy.pth` and evaluated against simple
opponents. On an M2 Max an iteration takes ≈ 40–50 s.

What to watch in TensorBoard:

| tag | meaning |
|---|---|
| `eval_current_strategy/*` | current CFR iterate (regret matching on the advantage nets) vs random / call / all-in, chips/hand, every 5 it. |
| `eval_avg_strategy/*` | quick fit of the *average* strategy — the thing CFR converges — every 25 it. |
| `advantage/seat*/final_loss` | the training loss; it is weighted by the iteration `t` (linear CFR) so it grows ~linearly by design |
| `advantage/seat*/mse_unweighted`, `target_rms` | unweighted fit error vs the scale of the sampled regrets (single-sample targets are very noisy, so the ratio stays high) |
| `samples/nodes_per_traversal` | game-tree size per traversal — grows as the bots stop folding |

Evaluate any policy (chips/hand, batched, native env):

```bash
python -m headsup.deepcfr.evaluate --policy cfr --hands 200000
python -m headsup.deepcfr.evaluate --policy cfr:runs/deepcfr/policy.pth --opponents random,call,allin,onnx
```

## Exploitability (best-response lower bound with rl_games)

`rl_games_env.py` registers `headsup_poker` with rl_games using our vectorised env (all
tables in one process, opponent inference batched / native in C++).

```bash
python exploitability.py -f rl_config/poker_env.yaml -t                        # train the exploiter (CPU)
python exploitability.py -f rl_config/poker_env.yaml -t --opponent cfr:runs/deepcfr/policy.pth
python exploitability.py -f rl_config/poker_env.yaml -p -c runs/<exp>/nn/exploitability.pth   # av reward = chips/hand
python rl_games_onnx.py -f rl_config/poker_env.yaml -m runs/<exp>/nn/exploitability.pth -o models/rl_games_exploiter.onnx
```

`--device mps` works too, but the exploiter's MLP is tiny and CPU is faster.

**About the old "500 mbb/g".** The original `poker_env.py` (used for the exploiter) had no
raise cap while the DeepCFR training env converted the 3rd consecutive raise into an all-in,
so the exploiter partly learned to raise repeatedly into lines the bot had never seen. With
uncapped raising the shipped exploiter wins ≈ +1.05 chips/hand (≈ 520 mbb/g) against the
shipped policy, but *loses* ≈ 0.67 chips/hand in the game the bot was trained on. Both sides
now share one engine; `env_config.raise_cap` selects the game (`1000000` = uncapped).

## Results

Chips/hand (1 chip = 500 mbb), 200 000 hands each:

| policy | vs random | vs call | vs all-in | PPO exploiter, 400 epochs, argmax |
|---|---|---|---|---|
| `models/deepcfr_policy.pth` (300 it., original run) | +3.61 | +4.72 | +2.27 | **+0.13** (≈ 66 mbb/g) |
| 100 it., `--value-steps 2000` (50 min on M2 Max) | +3.53 | +6.47 | +3.34 | +1.32 |

## Layout

```
headsup/engine.py        game rules, observation encoding, cheap clone()
headsup/env.py           PokerVecEnv / SingleAgentEnv (Python), NativeVecEnv (C++), rl_games IVecEnv API
headsup/players.py       batched players: random/call/allin/raise, torch policy, regret-matching iterate, numpy, onnx
headsup/model.py         DeepCFR network (torch); numpy_model.py = numpy mirror for CPU workers
headsup/cpp/             C++ kernels (pybind11): engine, evaluator, MLP, MCCFR traversal, VecEnv
headsup/deepcfr/         memory.py (reservoir), traverse.py, train.py, evaluate.py
headsup/web/             browser table: server.py (http.server), session.py (game logic), static/ (HTML/CSS/JS)
rl_games_env.py          rl_games registration;  exploitability.py / rl_games_onnx.py: exploiter tools
models/                  deepcfr_policy.pth, rl_games_exploiter.onnx
tests/                   pytest suite (rules, C++ ⇔ Python equivalence, envs, memories, web API)
```

## References

- N. Brown, A. Lerer, S. Gross, T. Sandholm — *Deep Counterfactual Regret Minimization* (ICML 2019)
- [treys](https://github.com/ihendley/treys) hand evaluator, [rl_games](https://github.com/Denys88/rl_games)
