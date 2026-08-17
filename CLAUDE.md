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
python -m pytest tests -q                                  # 60 tests, ~45 s; includes C++ <-> Python <-> numpy equivalence
```

- Device: automatic (`mps` > `cuda` > `cpu`); force with `--device` flags or `HEADSUP_DEVICE=cuda`.
  `torch.compile` is used for network fits (works on CUDA and MPS; falls back to eager).
- rl_games (exploiter) config `configs/rl_games_exploiter.yaml` sets `device: cpu`; on a CUDA box pass
  `--device cuda:0` to `exploitability.py` / `headsup.exploit` (tiny MLP — CPU is often faster anyway).
- macOS notes: build with Apple clang (setup.py forces `/usr/bin/clang` and strips a pyenv
  `-I<SDK>` flag that breaks libc++); MPS reductions over short strided dims are slow — avoided
  in `headsup/model.py` (do not "simplify" the elementwise adds back to `.sum(dim=1)`).
- Linux/CUDA: nothing special. C++ threads use all cores for traversals; the GPU does the fits,
  captured into CUDA graphs (`maybe_compile`: `reduce-overhead` on CUDA; ~2.7 ms/step vs 4.5).
  A 64-core / RTX 3090 box does ~25–30 s per iteration (40k traversals, 20 M memories); with two
  runs sharing the box (one per GPU) each takes ~35 s. Pass `--workers 56` (leave some cores).
- Rebuilding the extension while a trainer runs: build without `--inplace` and `mv` the .so into
  place (rename), never overwrite it in place - the running process has it mapped.
- C++ forward: `Linear::apply` is written as axpys over the transposed weight (contiguous inner loop,
  zero inputs skipped) so GCC vectorises it under strict FP: ~7 us per forward, ~110k traversal
  nodes/s per thread (was 36 us / 28k with the reduction form). Keep it that way; a 6-action game has
  ~230 nodes per traversal vs ~10-40 for the 4-action one.

## Repository map

```
headsup/engine.py        game rules + observation encoding (float32[80]); bet history; legal_mask(); cheap clone()
headsup/game.py          GameConfig (stack, blinds, raise_cap, bet_sizes, mask_redundant), raise amounts, legal masks/twins
headsup/cards.py         card ids 0..51 (rank = id % 13, suit = id // 13 in s,h,d,c order), treys tables, describe_hand()
headsup/env.py           PokerVecEnv / SingleAgentEnv (Python), NativeVecEnv (C++), make_vec_env(), play_hands()
headsup/players.py       batched players: random/call/allin/raise, TorchPolicyPlayer, RegretMatchingPlayer,
                         NumpyPolicyPlayer, ONNXPolicyPlayer, make_player(spec); mask_fold(); sample_actions()
headsup/model.py         BaseModel(features/arch/cards/dim/rm_fallback) - all variants; save() writes {config, state_dict};
                         numpy_model.py = numpy mirror; obs_dim_for(), bet_feature_indices() shared by all mirrors
headsup/cpp/headsup_cpp.cpp  pybind11: Engine, eval7 (treys-compatible), Model forward, run_traversals (MCCFR), VecEnv
headsup/native.py        loads the extension + installs treys lookup tables; native.available()
headsup/deepcfr/memory.py    ReservoirBuffer on the training device (uint8 card/stage part + float32 rest; obs_dim per model)
headsup/deepcfr/traverse.py  external-sampling traversal (Python reference) + TraversalRunner (C++ threads / Python procs)
headsup/deepcfr/train.py     trainer CLI (DeepCFR / SD-CFR / both), checkpoints, TensorBoard, final evaluation
headsup/deepcfr/evaluate.py  evaluate a player vs simple bots
headsup/sdcfr.py         IterateBank (all iterations' nets, vmapped) + SDCFRPlayer (exact / sample averaging, t^gamma weights)
headsup/compare.py       head-to-head round robin CLI;   headsup/exploit.py: K PPO exploiters -> exploitability lower bound
headsup/lbr.py           Local Best Response evaluator (CLI: python -m headsup.lbr --policy spec --hands N)
headsup/search.py        real-time search player: depth mode (SubgameSolver MCCFR + continuations, exact river) and
                         Pluribus mode @pluribus (VectorSolver: full remaining-game vector LCFR from the round start)
headsup/public.py        rebuild the public state (engine replay) from an observation row
headsup/blueprint.py     tabular Pluribus MCCFR-P blueprint (C++ TabularBlueprint; player spec tab:path.pt); trainer CLI
headsup/games/           Game interface for the generic algorithms: Kuhn/Leduc (leduc.py), hold'em presets (holdem.py)
headsup/algos/           tabular CFR family + MCCFR (tabular.py), exact best response (best_response.py),
                         DeepCFR/SD-CFR/DREAM/ESCHER on small games (deep.py), hold'em BR exploitability (holdem_br.py)
headsup/web/             browser table: server.py (stdlib http.server), session.py (game logic), static/ (HTML/CSS/JS)
headsup/rl/              rl_games integration: env.py (registration, own vec env, no Ray), exploitability.py (PPO CLI), onnx.py (export)
headsup/cpp/headsup_cpp.cpp  also: SubgameSolver, BoardTable, VectorSolver, PublicTree, Abstraction, TabularBlueprint
models/                  shipped models + README.json (retrain on the current rules/code before shipping)
tests/                   pytest suite (~115 tests, ~8 min on CPU; CI runs it on GitHub Actions)
```

## Game and data conventions (do not change without retraining everything)

- Seat 0 = dealer / small blind (acts first pre-flop), seat 1 = big blind (acts first post-flop).
  Blinds 1/2, stacks 100, reset every hand. Actions 0 fold, 1 check/call, 2..2+K-1 raise sizes
  (`GameConfig.bet_sizes`: default `("min",)` = call + 1 BB; pot fractions e.g. `(0.5, 1, 2)`), last =
  all-in; `num_actions = K + 3` (max 8, `MAX_ACTIONS` in game.py and the C++). raise_cap-th raise in a row
  -> all-in (`raise_cap=3`). **Fold only exists when facing a bet**; with nothing to call FOLD is
  executed as CHECK. `engine.legal_mask()` (Python + C++) == `engine.legal_mask_from_obs(obs, game)`:
  fold-when-facing-a-bet, and with `mask_redundant` (auto-on for custom sizes, OFF for the default
  game to keep its historic tree) also no raise that duplicates another action; traversals skip
  masked actions (their advantage target = the twin's), regret matching / players (`players.mask_illegal`)
  zero them. Reward = chips won (1 chip = 500 mbb). The tree (bet_sizes, raise_cap, mask_redundant) is
  in every model config (`config["game"]`); envs take it from network players (`resolve_game`), so pass
  `game=agent.game` to `make_vec_env` when the agent is a network and the opponent a simple bot.
- Observation `float32[80]` (OBS_DIM): [0:6] hand 2 x (rank+1, suit+1, card+1) sorted by id; [6:21] board
  5 x same (flop sorted; 0 = no card); [21] stage 0..3; [22] 1 if big blind (first to act post-flop);
  [23] (opp street bet - own) / pot (0 exactly when nothing to call); [24] own total bet / pot; [25] opp
  total bet / pot; [26] own street bet / pot; [27] opp street bet / pot; [28] stack / pot; [29] pot / 1000;
  [30] to-call / stack; [31:79] bet history (DeepCFR paper): 4 rounds x 6 slots x [chips / pot before the
  action, occurred]; [79] consecutive raises this street (RAISES_INDEX; for legal_mask_from_obs).
  Networks read a prefix: `features=aggregated` -> obs[:31] (OBS_DIM_AGGREGATED), history/both -> obs[:79]
  (OBS_DIM_HISTORY). Envs always hand out the full vector; `BaseModel.forward` / NumpyModel / C++ Model slice.
  Card ids: `id = rank + 13*suit`, ranks 2..A = 0..12, suits s,h,d,c = 0..3.
- Model config (`headsup.model.DEFAULT_CONFIG`): features aggregated|history|both, arch current|paper,
  cards embed|onehot, dim, rm_fallback uniform|argmax. Saved inside policy.pth ({config, state_dict}),
  iterates.pt ("config"), checkpoints ("model_config") and `numpy_weights()` ("config"); there is no
  legacy loading - old raw state dicts / config-less banks are not supported.
- Player protocol: `player(obs[N,80], ids=None) -> int64[N]`; optional `probs(obs, ids)`,
  `last_probs`, `game` (GameConfig of the tree it plays); stateful players set `wants_ids = True`
  (SD-CFR tracks per-table reach in dense tensors indexed by id).
- Player specs (CLI/UI/yaml): `cfr[:path.pth]`, `sdcfr:path/iterates.pt[@exact|@sample][@g<gamma>][@t<N>][@k<K>]`
  (`@t100` = average after 100 iterations, `@k64` = bank thinned to 64 representative iterates),
  `iterate:path/iterates.pt[@t<N>]` (the current strategy of iteration N), `onnx[:path.onnx]`, `random`,
  `call`, `allin`, `raise`. ONNX exploiters read the input width they were exported with.
- Vec envs alternate seats between hands; when the opponent open-folds during reset, the reset
  observation is terminal and the next step (any action) collects the reward.

## Algorithms (see README "How close is this to the papers?")

- DeepCFR: external sampling; advantage nets re-initialised and fitted from scratch each iteration
  (loss `mean(t * (pred - target)^2)`, Adam 1e-3, grad clip 1, `--value-steps 4000 x --batch-size 16384`);
  reservoir memories (`--adv-capacity`, `--strat-capacity`, default 10 M); average-strategy net fitted
  on the strategy memory with t-weighted MSE (`--policy-epochs 50`, or `--policy-steps N`). Regret
  matching falls back to uniform (over allowed actions) when no advantage is positive, or to the
  highest advantage with `--rm-fallback argmax` (paper). `--preset paper` = paper hyperparameters.
- Paper facts verified (arXiv 1811.00164 / 1901.07621): bet history = 6 slots x rounds x [size, occurred];
  RM fallback = argmax; K = 10k traversals, batch 10k, 4000 steps, 40 M memories; SD-CFR one-hot cards,
  avg-strategy net 20k x 20480, 300k traversals/it on 5-FHP. The "98,948 parameters" of the DeepCFR
  paper is not reproducible from its Appendix C code at any integer width (dim 64 -> 66,499 for HULH).
- SD-CFR: `--algo sdcfr|both` keeps every iteration's advantage nets (`iterates.pt`, ~0.5 MB/iteration)
  and `SDCFRPlayer` plays the exact reach-weighted linear average or per-hand trajectory sampling
  (same action-sequence distribution; the telescoping identity is a test). Weights t^gamma (`@g2`).
- The advantage-net loss grows ~linearly with iterations by construction (t weighting): watch
  `eval_current_strategy/*`, `eval_avg_strategy/*`, `eval_sdcfr/*` and `advantage/*/mse_unweighted` instead.

## Everyday commands

```bash
python -m headsup.deepcfr.train --algo both --iterations 300 --traversals 40000 --checkpoint-every 20 --workers 56 --out runs/x
python -m headsup.deepcfr.train --features history --net paper --rm-fallback argmax --preset paper --out runs/paper  # paper-faithful
python -m headsup.deepcfr.train --resume runs/x/checkpoint.pt --iterations 300 --out runs/x     # continue
python -m headsup.deepcfr.train --policy-only runs/x/checkpoint.pt --out runs/x               # refit artefacts + eval
tensorboard --logdir runs
python -m headsup.deepcfr.evaluate --policy cfr:runs/x/policy.pth --hands 200000
python -m headsup.compare cfr:runs/x/policy.pth sdcfr:runs/x/iterates.pt cfr --hands 1000000 --bots
python -m headsup.exploit --policy sdcfr:runs/x/iterates.pt --seeds 3 --epochs 1000 --hands 400000 --parallel 3
python -m headsup.lbr --policy sdcfr:runs/x/iterates.pt --hands 20000 --model-iterates 32     # LBR bound, duplicate pairs
python -m headsup.web --opponent sdcfr:runs/x/iterates.pt --advisor cfr --port 8000
python -m headsup.lbr --policy cfr:runs/x/policy.pth --hands 30000 --num-tables 128 --workers 24
python -m headsup.blueprint --game nlhe --iterations 20000000 --threads 24 --out runs/bp/nlhe.pt   # tabular blueprint
python -m headsup.web --opponent search:tab:runs/bp/nlhe.pt@pluribus@th16                            # Pluribus-style bot
python -m headsup.algos.holdem_br --policy cfr:runs/fhp/policy.pth --game fhp --boards 200           # FHP/HULH exploitability
python -m headsup.algos.deep --game leduc --algo dream --iterations 200 --eval-every 10               # small-game algorithms
```

Player specs also: ``search:<blueprint>[@it..][@pluribus][@th<threads>]``, ``tab:path.pt``, ``iterate:iterates.pt[@t<N>]``.
Rebuild the extension with ``python setup.py build_ext`` and ``mv`` the .so into place (never ``--inplace``
while trainers run: they map the file).

Timings on an M2 Max (12 cores, MPS): traversals ~20k/s (C++ threads), 4000-step fit ~18 s,
one iteration 40–55 s at 40k traversals; 300 iterations ~4.5 h; `headsup.exploit` 3 seeds x 1000
epochs ~30 min in parallel; the final 50-epoch policy fit ~3 min. Checkpoints with 10 M memories are
~4 GB (`--checkpoint-every`), `runs/` is git-ignored — clean it up after experiments.

## Results so far (all measured with this code; details in README "Results" and models/README.json)

- Shipped `models/deepcfr_policy.pth` = `runs/abl_history_paper` (history features, paper net, 300 it. x 40k
  traversals, new rules): +4.3 / +7.9 / +3.3 chips/hand vs random / call / all-in; LBR 1.33 +- 0.09; PPO
  exploiter 0.01; best-response exploitability (holdem_br, 12 boards) 3.24 chips = 1620 mbb/g. All DeepCFR
  arms tie head-to-head with their SD-CFR averages (+-0.05 at 1.5 M hands).
- Shipped `models/blueprint_nlhe.pt` = tabular MCCFR-P blueprint (20 M it., 200 EHS buckets, 40 min): LBR
  0.67 +- 0.09; beats the DeepCFR policy +0.81 +- 0.04 and its SD-CFR average +0.82 head-to-head (300k
  hands) and its own 400k-iteration version +0.33 +- 0.09; scores less vs fixed bots (+1.8 / +2.5 / +0.9) -
  equilibria don't exploit. Table-mode abstraction trains at ~60k it/s once warm.
- Exploiter power: PPO << LBR << BR (BR finds ~2.5x LBR); use `holdem_br` for exploitability claims.
- FHP: EHS-bucket tabular blueprints floor at ~380 mbb/g regardless of buckets / iterations (1-D
  abstraction floor; cf. DeepCFR paper Fig. 2 abstraction baselines). Leduc (1 seed, 200 it., mA/g):
  SD-CFR 270, DeepCFR 307, DREAM 385, ESCHER 528; tabular CFR+ 35 / DCFR 32 at 1000 it.
- In flight (2026-08-17): long_history_paper (1000 it.), FHP DeepCFR / DREAM / ESCHER reproductions,
  betsize (0.5/1/2 pot) DeepCFR vs tabular, search-player LBRs (`runs/bp/eval.log`, `runs/queue.log`).

## Working agreements

- Never drive `/api/*` of a web server a human is playing on (it is their session); use another port.
- Keep C++ and Python behaviour identical; every engine/traversal change needs both implementations
  and the differential tests. Rebuild the extension after touching `headsup/cpp/`.
- Commit/push only when asked; results changes go with README + `models/README.json` updates.
- Search: pre-river subgames use *sampled* MCCFR where LCFR beats DCFR/CFR+/PCFR+ (measured); the
  river uses full-width vector CFR where DCFR/CFR+/PCFR+ win. Do not "improve" the sampled solver with
  regret flooring. `search:` players are stateful (wants_ids) and slow (~1-2 s/decision): evaluate on
  1-2k hands, LBR models them with the blueprint (still a valid lower bound).
- Prefer measuring (compare / exploit / lbr tools, 400k+ hands; LBR 20k+ duplicate pairs) over anecdotes;
  report ± standard errors. LBR vs an SD-CFR bank: query cost ~4 ms x iterates per 64-table query, so
  thin the queried bank (`--model-iterates 32`); the bound stays valid (LBR only gets weaker).
- Observations use a canonical card order (hole cards and flop sorted by id) - required by LBR's hand
  substitution and by the one-hot card variant; embedding nets are order-invariant anyway.
