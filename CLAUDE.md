# CLAUDE.md — working notes for this repository

Heads-up no-limit-style Texas Hold'em: game engine (Python + C++), DeepCFR / Single Deep CFR
training, exploitability tools, browser UI. Everything below is what a new session needs to
know to run, extend and evaluate the project on any machine (Linux/CUDA, macOS/MPS, CPU).

## Setup on a fresh machine

```bash
git clone https://github.com/romaf5/headsup-poker && cd headsup-poker
python3.11 -m venv .venv && source .venv/bin/activate     # 3.10–3.12 work; 3.11 tested
pip install -r requirements.txt                            # torch, numpy, treys, tqdm, tensorboard, pybind11, pytest
pip install -r requirements-rl.txt                         # + rl_games, gymnasium, onnx, onnxruntime (exploiter tools)
python setup.py build_ext --inplace                        # C++ kernels -> headsup_cpp.*.so (needs a C++17 compiler)
python -m pytest tests -q                                  # 39 tests, ~25 s; includes C++ <-> Python equivalence
```

- Device: automatic (`mps` > `cuda` > `cpu`); force with `--device` flags or `HEADSUP_DEVICE=cuda`.
  `torch.compile` is used for network fits (works on CUDA and MPS; falls back to eager).
- rl_games (exploiter) config `configs/rl_games_exploiter.yaml` sets `device: cpu`; on a CUDA box pass
  `--device cuda:0` to `exploitability.py` / `headsup.exploit` (tiny MLP — CPU is often faster anyway).
- macOS notes: build with Apple clang (setup.py forces `/usr/bin/clang` and strips a pyenv
  `-I<SDK>` flag that breaks libc++); MPS reductions over short strided dims are slow — avoided
  in `headsup/model.py` (do not "simplify" the elementwise adds back to `.sum(dim=1)`).
- Linux/CUDA: nothing special. C++ threads use all cores for traversals; the GPU does the fits.
  Expect ~2–4x faster iterations than the M2 Max numbers quoted below.

## Repository map

```
headsup/engine.py        game rules + observation encoding (float32[31]); cheap clone(); fold_allowed_mask()
headsup/cards.py         card ids 0..51 (rank = id % 13, suit = id // 13 in s,h,d,c order), treys tables, describe_hand()
headsup/env.py           PokerVecEnv / SingleAgentEnv (Python), NativeVecEnv (C++), make_vec_env(), play_hands()
headsup/players.py       batched players: random/call/allin/raise, TorchPolicyPlayer, RegretMatchingPlayer,
                         NumpyPolicyPlayer, ONNXPolicyPlayer, make_player(spec); mask_fold(); sample_actions()
headsup/model.py         BaseModel (DeepCFR net, flat obs in, 4 logits/advantages out); numpy_model.py = numpy mirror
headsup/cpp/headsup_cpp.cpp  pybind11: Engine, eval7 (treys-compatible), Model forward, run_traversals (MCCFR), VecEnv
headsup/native.py        loads the extension + installs treys lookup tables; native.available()
headsup/deepcfr/memory.py    ReservoirBuffer on the training device
headsup/deepcfr/traverse.py  external-sampling traversal (Python reference) + TraversalRunner (C++ threads / Python procs)
headsup/deepcfr/train.py     trainer CLI (DeepCFR / SD-CFR / both), checkpoints, TensorBoard, final evaluation
headsup/deepcfr/evaluate.py  evaluate a player vs simple bots
headsup/sdcfr.py         IterateBank (all iterations' nets, vmapped) + SDCFRPlayer (exact / sample averaging, t^gamma weights)
headsup/compare.py       head-to-head round robin CLI;   headsup/exploit.py: K PPO exploiters -> exploitability lower bound
headsup/web/             browser table: server.py (stdlib http.server), session.py (game logic), static/ (HTML/CSS/JS)
headsup/rl/              rl_games integration: env.py (registration, own vec env, no Ray), exploitability.py (PPO CLI), onnx.py (export)
models/                  deepcfr_policy.pth (shipped), deepcfr_policy_v1.pth (original), rl_games_exploiter.onnx, README.json
tests/                   pytest suite (rules, golden observations, C++ <-> Python, envs, memories, SD-CFR, web API)
```

## Game and data conventions (do not change without retraining everything)

- Seat 0 = dealer / small blind (acts first pre-flop), seat 1 = big blind (acts first post-flop).
  Blinds 1/2, stacks 100, reset every hand. Actions 0 fold, 1 check/call, 2 min-raise (call + 1 BB),
  3 all-in. 3rd raise in a row -> all-in (`raise_cap=3`). **Fold only exists when facing a bet**
  (`engine.fold_allowed`); with nothing to call FOLD is executed as CHECK, traversals skip it and
  regret matching / all players mask it (`players.mask_fold`). Reward = chips won (1 chip = 500 mbb).
- Observation `float32[31]`: [0:6] hand 2 x (rank+1, suit+1, card+1); [6:21] board 5 x same (0 = no
  card); [21] stage 0..3; [22] 1 if big blind (first to act post-flop); [23] (opp street bet - own) / pot
  (0 exactly when nothing to call); [24] own total bet / pot; [25] opp total bet / pot; [26] own
  street bet / pot; [27] opp street bet / pot; [28] stack / pot; [29] pot / 1000; [30] to-call / stack.
  Card ids: `id = rank + 13*suit`, ranks 2..A = 0..12, suits s,h,d,c = 0..3 (compatible with old models).
- Player protocol: `player(obs[N,31], ids=None) -> int64[N]`; optional `probs(obs, ids)`,
  `last_probs`; stateful players set `wants_ids = True` (SD-CFR tracks per-table reach).
- Player specs (CLI/UI/yaml): `cfr[:path.pth]`, `sdcfr:path/iterates.pt[@exact|@sample][@g<gamma>]`,
  `onnx[:path.onnx]`, `random`, `call`, `allin`, `raise`.
- Vec envs alternate seats between hands; when the opponent open-folds during reset, the reset
  observation is terminal and the next step (any action) collects the reward.

## Algorithms (see README "How close is this to the papers?")

- DeepCFR: external sampling; advantage nets re-initialised and fitted from scratch each iteration
  (loss `mean(t * (pred - target)^2)`, Adam 1e-3, grad clip 1, `--value-steps 4000 x --batch-size 16384`);
  reservoir memories (`--adv-capacity`, `--strat-capacity`, default 10 M); average-strategy net fitted
  on the strategy memory with t-weighted MSE (`--policy-epochs 50`). Regret matching falls back to
  uniform (over allowed actions) when no advantage is positive.
- SD-CFR: `--algo sdcfr|both` keeps every iteration's advantage nets (`iterates.pt`, ~0.5 MB/iteration)
  and `SDCFRPlayer` plays the exact reach-weighted linear average or per-hand trajectory sampling
  (same action-sequence distribution; the telescoping identity is a test). Weights t^gamma (`@g2`).
- The advantage-net loss grows ~linearly with iterations by construction (t weighting): watch
  `eval_current_strategy/*`, `eval_avg_strategy/*`, `eval_sdcfr/*` and `advantage/*/mse_unweighted` instead.

## Everyday commands

```bash
python -m headsup.deepcfr.train --algo both --iterations 300 --traversals 40000 --checkpoint-every 10 --out runs/x
python -m headsup.deepcfr.train --resume runs/x/checkpoint.pt --iterations 300 --out runs/x     # continue
python -m headsup.deepcfr.train --policy-only runs/x/checkpoint.pt --out runs/x               # refit artefacts + eval
tensorboard --logdir runs
python -m headsup.deepcfr.evaluate --policy cfr:runs/x/policy.pth --hands 200000
python -m headsup.compare cfr:runs/x/policy.pth sdcfr:runs/x/iterates.pt cfr --hands 1000000 --bots
python -m headsup.exploit --policy sdcfr:runs/x/iterates.pt --seeds 3 --epochs 1000 --hands 400000 --parallel 3
python -m headsup.web --opponent sdcfr:runs/x/iterates.pt --advisor cfr --port 8000
```

Timings on an M2 Max (12 cores, MPS): traversals ~20k/s (C++ threads), 4000-step fit ~18 s,
one iteration 40–55 s at 40k traversals; 300 iterations ~4.5 h; `headsup.exploit` 3 seeds x 1000
epochs ~30 min in parallel; the final 50-epoch policy fit ~3 min. Checkpoints with 10 M memories are
~4 GB (`--checkpoint-every`), `runs/` is git-ignored — clean it up after experiments.

## Results so far (all measured with this code; details in models/README.json)

- Shipped `models/deepcfr_policy.pth` (300 it., 10k->40k traversals, old rules): +3.7 / +6.3 / +3.0
  chips/hand vs random / call / all-in; best PPO exploiter +0.31 (old rules) -> +0.09 with the
  no-free-fold mask; head-to-head tie with the original v1 model.
- Finding: the average strategies (policy net and exact SD-CFR average alike) folded ~17 % of flops
  for free before the rule change; the latest iterate only 1 % — early iterations' weight in the
  average. Hence the rule change; retrain everything on the new rules before comparing to old numbers.
- Under the fixed rules a run was started (`--algo both`, 300 it., 40k traversals) and stopped at
  it. 52 to move to a faster machine; at it. 50 both averages were at ~+4.5 / +8 / +3 vs the bots.

## Working agreements

- Never drive `/api/*` of a web server a human is playing on (it is their session); use another port.
- Keep C++ and Python behaviour identical; every engine/traversal change needs both implementations
  and the differential tests. Rebuild the extension after touching `headsup/cpp/`.
- Commit/push only when asked; results changes go with README + `models/README.json` updates.
- Prefer measuring (compare / exploit tools, 400k+ hands) over anecdotes; report ± standard errors.
