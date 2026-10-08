# AlphaHoldem Implementation Plan

**Goal:** AlphaHoldem (Zhao et al., AAAI-22) for the default no-limit game: card / action tensors, pseudo-siamese
network, Trinal-Clip PPO, K-Best self-play, a trainer and a player spec.

**Architecture:** a two-seat vectorised env (`headsup_cpp.SelfPlayVecEnv` + the Python twin in `headsup/twoseat.py`)
and the package `headsup/alphaholdem/` (`encoding.py`, `model.py`, `ppo.py`, `pool.py`, `train.py`, `player.py`).

**Spec:** `docs/superpowers/specs/2026-10-08-alphaholdem-design.md`

## Global constraints

- Python is `/home/mario/workdir/headsup-poker/.venv/bin/python`, run from the worktree; CPU only
  (`CUDA_VISIBLE_DEVICES=""`), at most 8 threads, no command over ~10 minutes.
- Existing files change by small additions only: the C++ struct after `VecEnv`, its bindings at the end of the
  module, one branch in `make_player`, one label in the browser table, one README row.
- The extension is built inside the worktree (`python setup.py build_ext --inplace`).
- Tests first; the two new test files run in under a minute together on CPU.
- Commit messages end with exactly `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Tasks

1. **Two-seat env.** Tests (`tests/test_twoseat.py`): C++ against the twin on shared decks with random actions and
   waits, with and without all-in EV; the twin against `PokerVecEnv` (same seed, deterministic players: the same
   rewards per hand); invalid actions. Then `TwoSeatVecEnv` (Python), `SelfPlayVecEnv` (C++), the wrapper and
   `make_two_seat_env`.
2. **Encoding.** Tests: tensors against the engine's true history in three trees (default; five pot-fraction sizes
   with `mask_redundant`; two sizes, cap 4), card tensor, legal mask against `legal_mask_from_obs`, value bounds,
   terminal observations. Then `Encoder`.
3. **Network.** Tests: shapes, parameter split, illegal logits masked, towers do not share parameters, save / load.
   Then `AlphaNet`.
4. **Loss and advantages.** Tests: the three clips by hand (values and gradients), the value target, GAE on a
   hand-built trajectory. Then `trinal_clip_objective`, `clipped_value_target`, `stream_gae`, `ppo_loss`.
5. **Pool.** Tests: ELO update, K-best eviction, balanced opponent assignment. Then `KBestPool`.
6. **Trainer.** Tests: rollout invariants (main-controlled seats only, complete hands, returns = rewards), frozen
   opponents, the main agent moves, checkpoint round trip, short run. Then `Trainer` and the CLI.
7. **Player.** Tests: probabilities, legality, batch consistency, the spec, `compare` / LBR accept it. Then
   `AlphaHoldemPlayer`, `alpha:` in `make_player`, the browser label.
8. **Mutation check.** One-line mutations of the wiring (listed below with the test that must fail).
9. **Measurements.** Env steps/s, PPO samples/s on CPU, the CPU smoke run; README row; the whole suite.

## Mutation check (filled in at the end)

See the table appended by task 8.
