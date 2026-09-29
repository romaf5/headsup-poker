# CLAUDE.md — working notes for this repository

Heads-up hold'em research code: engine (Python + C++), DeepCFR / SD-CFR / DREAM / ESCHER, tabular
Pluribus-style blueprint + real-time search, best-response / LBR / PPO evaluation, browser UI.
README.md is the user-facing documentation (commands, results); this file holds what a new
session needs to work on the code safely.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate       # 3.10-3.12
pip install -r requirements.txt -r requirements-rl.txt
python setup.py build_ext --inplace                          # C++17; -> headsup_cpp.*.so
python -m pytest tests -q                                    # ~115 tests, ~8 min on CPU (CI: GitHub Actions)
```

- Device: automatic (`mps` > `cuda` > `cpu`), `--device` / `HEADSUP_DEVICE=cuda:0` to force. CPU-only
  jobs: `CUDA_VISIBLE_DEVICES=""`. The rl_games exploiter config uses `device: cpu` (tiny MLP).
- **Never rebuild the extension in place while a process uses it**: `python setup.py build_ext` then
  `mv` the new .so over the old one (running processes have the file mapped).
- macOS: setup.py forces Apple clang; MPS reductions over short strided dims are slow — keep the
  elementwise adds in `headsup/model.py` (do not "simplify" them to `.sum(dim=1)`).
- C++ `Linear::apply` is written as axpys over the transposed weight (vectorises under strict FP,
  ~7 us per forward); keep it that way.

## Conventions (changing any of these invalidates trained models)

- Seat 0 = dealer / small blind (acts first pre-flop), seat 1 = big blind (first post-flop).
  Actions: 0 fold, 1 check/call, 2..2+K-1 raise sizes (`GameConfig.bet_sizes`), last = all-in
  (NL); `num_actions = K + 3` (max 8). Fold only exists when facing a bet (else it is executed as a
  check). `engine.legal_mask()` (Python + C++) == `legal_mask_from_obs(obs, game)`; with
  `mask_redundant` (auto-on for custom sizes) raises duplicating another action are masked and
  traversals give them their twin's target. Limit games (FHP / HULH): `limit` raise increments,
  `raise_caps` per round, `num_rounds`, `all_in=False`, 100 000-chip stacks.
- Observation `float32[80]`: [0:6] hand, [6:21] board (rank+1, suit+1, card+1; hole cards and flop
  sorted by id — LBR's hand substitution and one-hot cards rely on it), [21] stage, [22] big blind,
  [23:31] pot-normalised bet / stack features (stack capped at 1000 chips), [31:79] bet history
  (4 rounds × 6 slots × [chips / pot before the action, occurred]), [79] consecutive raises. Networks
  read obs[:31] (`features=aggregated`) or obs[:79] (`history` / `both`); envs always hand out 80.
  Card id = rank + 13·suit (ranks 2..A = 0..12, suits s h d c).
- Every model artefact stores its config incl. the action tree (`config["game"]`, with blinds /
  stacks when not the NL defaults); there is no legacy loading. Envs take the game from network
  players (`resolve_game`) — pass `game=agent.game` to `make_vec_env` when the opponent is a bot.
- Player protocol: `player(obs[N,80], ids=None) -> int64[N]`, optional `probs(obs, ids)`, `game`;
  stateful players (SD-CFR sampling, search) set `wants_ids = True`. Specs: see README "Evaluation".
- Vec envs alternate seats between hands; when the opponent open-folds during reset, the reset
  observation is terminal and the next step collects the reward.
- Keep C++ and Python behaviour identical: every engine / traversal / network change needs both
  implementations and the differential tests.

## Algorithm notes

- DeepCFR advantage loss `mean(t · (pred − target)²)` grows ~linearly with t by construction; watch
  `eval_*` and `advantage/*/mse_unweighted` instead. `--preset paper` = paper hyperparameters.
- The paper's "98 948 parameters" is not reproducible from its Appendix C code at any integer width.
- Search: pre-river subgames use *sampled* MCCFR where LCFR beats DCFR / CFR+ / PCFR+ (measured); the
  river uses full-width vector CFR where DCFR / CFR+ / PCFR+ win. Do not add regret flooring to the
  sampled solver. `search:` players are slow (~1-2 s/decision): evaluate on 1-2k hands.
- Exploitability claims: use `headsup.algos.holdem_br` (GPU; FHP exact with `--cards all`; papers differ on
  mean-over-seats vs total exploitability - DeepCFR's FHP numbers are totals). LBR vs an
  SD-CFR bank: thin the queried bank (`--model-iterates 32`; the bound stays valid).
- Measure, don't anecdote: 400k+ hands for head-to-head, 20k+ LBR pairs, report ± SE.

## Machine / operations (64-core WSL2 box, 2 × RTX 3090, 94 GB RAM)

- GPU 1 may be used by another project (check `nvidia-smi` and the processes' `CUDA_VISIBLE_DEVICES`
  before launching); use the free GPU with `CUDA_VISIBLE_DEVICES=<n> ... --device cuda:0`.
- Host RAM: the WSL2 VM is capped at 96 GB and has rebooted under memory pressure. Keep total RSS
  < ~50 GB: at most one `--memory-device cpu` trainer per GPU, checkpoints every 25-50 iterations
  (10-20 M memories ≈ 4-16 GB per checkpoint), `MALLOC_ARENA_MAX=2-4` for many-thread solvers.
- Use ~56 of 64 cores (`--workers 56`); `runs/` is git-ignored (clean it up after experiments).
- Never drive `/api/*` of a web server a human is playing on; use another port.
- `pgrep -f` / `pkill -f` match the launching shell too — wait on pid files / `kill -0 $PID`.
- Commit / push to `main` when asked; result changes go with README + `models/README.json`.
