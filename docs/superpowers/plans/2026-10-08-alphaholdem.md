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

All tasks are done; results and what is not verified are in the spec.

## Mutation check

Each row is a one-line change of the source; the named test fails on it (59 of 59; for the C++ rows the extension
was rebuilt). Two rows were not caught at first: "the seat change does not reach the actor selection" - on CPU the
seat tensor shared memory with the numpy array, so the refresh was only needed on a GPU; the tensor is a copy on
every device now - and "the opponent-assignment generator not restored" - the test compared two resumed trainers
with each other; it now compares both generators with the saved states. "waiting tables are woken up" makes the
rollout loop endless; the test does not return, which counts as failing.

| mutation | test that fails |
|---|---|
| twin: the seats' rewards swapped | `test_twin_plays_the_hands_of_the_one_seat_env` |
| twin: a waiting table checks instead | `test_twin_turn_order_rewards_and_waiting` |
| twin: decks ignored | `test_twin_takes_decks_for_the_next_deal` |
| encoding: seat rows swapped | `test_action_tensor_matches_the_true_history` |
| encoding: past legal rows all ones | `test_action_tensor_matches_the_true_history` |
| encoding: no pending legal row | `test_action_tensor_matches_the_true_history` |
| encoding: all-ins not recognised | `test_action_tensor_matches_the_true_history` |
| encoding: a call of an all-in recorded as all-in | `test_action_tensor_matches_the_true_history` |
| encoding: raise counter of past decisions always 0 | `test_action_tensor_matches_the_true_history` |
| encoding: the small blind's posted chips wrong | `test_action_tensor_matches_the_true_history` |
| encoding: turn and river channels swapped | `test_card_tensor_and_value_bounds` |
| encoding: own / opponent chips swapped | `test_card_tensor_and_value_bounds` |
| encoding: raise cap off by one | `test_legal_mask_equals_the_engines` |
| network: illegal logits not masked | `test_player_probabilities_legality_and_batching` |
| network: the action tensor does not reach the heads | `test_network_shapes_masking_and_separate_towers` |
| loss: no delta1 clip (plain PPO) | `test_trinal_clip_policy_objective_by_hand` |
| loss: min over three terms (the re-implementation's form) | `test_trinal_clip_policy_objective_by_hand` |
| loss: delta1 clip on the wrong sign | `test_trinal_clip_policy_objective_by_hand` |
| loss: value bounds swapped | `test_value_target_is_clipped_to_the_chips_put_in_so_far` |
| loss: value target not clipped | `test_ppo_loss_puts_the_pieces_together` |
| loss: entropy sign | `test_ppo_loss_puts_the_pieces_together` |
| loss: policy sign | `test_ppo_loss_puts_the_pieces_together` |
| GAE: the reward is credited to every decision of the hand | `test_gae_along_seat_streams_by_hand` |
| GAE: a reward without a decision leaks into the previous hand | `test_gae_along_seat_streams_by_hand` |
| GAE: values carried across hands | `test_gae_along_seat_streams_by_hand` |
| GAE: waiting cells enter the stream | `test_gae_along_seat_streams_by_hand` |
| GAE: the other seat's reward | `test_gae_along_seat_streams_by_hand` |
| pool: the member is the live network | `test_training_moves_the_main_agent_and_not_the_pool` |
| pool: the member's rating moves with the main agent's | `test_elo_update_and_k_best_selection` |
| pool: the best member leaves | `test_elo_update_and_k_best_selection` |
| pool: a won block counts as lost | `test_elo_game_follows_the_rollout_result` |
| pool: the current agent is never an opponent | `test_pool_assigns_opponents_evenly_and_round_trips` |
| trainer: the opponent's decisions are samples too | `test_rollout_trains_on_the_main_agents_decisions_only` |
| trainer: the main agent keeps its seat | `test_rollout_trains_on_the_main_agents_decisions_only` |
| trainer: the seat change does not reach the actor selection | `test_rollout_samples_are_aligned_with_their_observations` |
| trainer: the pool result is the opponent's | `test_rollout_trains_on_the_main_agents_decisions_only` |
| trainer: the main network plays the pool member's seat | `test_rollout_trains_on_the_main_agents_decisions_only` |
| trainer: rewards not scaled | `test_rollout_trains_on_the_main_agents_decisions_only` |
| trainer: the rollout stops in the middle of hands | `test_rollout_returns_are_each_seats_reward_of_its_hand` |
| trainer: waiting tables are woken up | `test_rollout_returns_are_each_seats_reward_of_its_hand` |
| trainer: value bounds of other tables | `test_rollout_samples_are_aligned_with_their_observations` |
| trainer: the stored seat is the other one | `test_rollout_samples_are_aligned_with_their_observations` |
| trainer: values missing from the grid (GAE sees zeros) | `test_rollout_samples_are_aligned_with_their_observations` |
| trainer: advantages not normalised | `test_update_uses_the_configured_loss` |
| trainer: value and entropy coefficients swapped | `test_update_uses_the_configured_loss` |
| trainer: delta1 not taken from the settings | `test_update_uses_the_configured_loss` |
| trainer: --no-value-clip ignored | `test_update_uses_the_configured_loss` |
| trainer: no optimiser step | `test_training_moves_the_main_agent_and_not_the_pool` |
| trainer: the ELO game gets the opponent's chips | `test_elo_game_follows_the_rollout_result` |
| trainer: Adam's state not restored | `test_checkpoint_round_trip_and_cli` |
| trainer: the pool not restored | `test_checkpoint_round_trip_and_cli` |
| trainer: the opponent-assignment generator not restored | `test_checkpoint_round_trip_and_cli` |
| trainer: the action-sampling generator not restored | `test_checkpoint_round_trip_and_cli` |
| player: always the first action | `test_player_probabilities_legality_and_batching` |
| spec: deterministic not passed on | `test_player_probabilities_legality_and_batching` |
| C++ env: the seats' rewards swapped | `test_cpp_env_matches_the_python_twin` |
| C++ env: another all-in EV seed | `test_cpp_env_matches_the_python_twin` |
| C++ env: a waiting table checks instead | `test_cpp_env_matches_the_python_twin` |
| C++ env: the reported seat is the other one | `test_cpp_env_matches_the_python_twin` |
