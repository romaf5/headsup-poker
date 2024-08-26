import psutil
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from model import BaseModel
from bounded_storage import BoundedStorage, GPUBoundedStorage
from player_wrapper import PolicyPlayerWrapper
from pokerenv_cfr import Action, HeadsUpPoker, ObsProcessor
from cfr_env_wrapper import CFREnvWrapper
from eval_policy import EvalPolicyPlayer

from time_tools import Timers

NUM_WORKERS = 16
BOUNDED_STORAGE_MAX_SIZE = 40_000_000


class EvalPolicy:
    def __init__(self, env, policy, logger):
        self.env = env
        self.player = PolicyPlayerWrapper(policy)
        self.logger = logger

    def eval(self, games_to_play=50000):
        eval_policy_player = EvalPolicyPlayer(self.env)
        simple_player_scores = eval_policy_player.eval(self.player, games_to_play)

        for opponent_name, score in simple_player_scores.items():
            self.logger.add_scalar(
                f"policy_evaluation/{opponent_name}/mean_reward", score
            )


def _batch_obses(obses):
    return {k: torch.tensor([obs[k] for obs in obses]) for k in obses[0].keys()}


class BatchSampler:
    def __init__(self, samples):
        self.dicts = [sample[0] for sample in samples]
        self.ts = [sample[1] for sample in samples]
        self.values = [sample[2] for sample in samples]

        # reshape dicts
        keys = self.dicts[0].keys()
        self.dicts = {
            k: torch.tensor([dct[k] for dct in self.dicts], device="cuda") for k in keys
        }
        self.ts = torch.tensor(self.ts, device="cuda")[..., None]
        self.values = torch.tensor(np.array(self.values), device="cuda")

    def __len__(self):
        return len(self.ts)

    def __call__(self, batch_size):
        indices = np.random.choice(len(self), batch_size)

        obs = {k: v[indices] for k, v in self.dicts.items()}
        ts = self.ts[indices]
        values = self.values[indices]

        return obs, ts, values


class BatchSamplerGPUStorage:
    def __init__(self, bounded_storage):
        self.dicts, self.ts, self.values = bounded_storage.get_storage()

    def __len__(self):
        return len(self.ts)

    def __call__(self, batch_size):
        indices = np.random.choice(len(self), batch_size)

        obs = {k: v[indices] for k, v in self.dicts.items()}
        ts = self.ts[indices]
        values = self.values[indices]

        return obs, ts, values


def train_values(player, samples):
    mini_batches = 4000
    optimizer = torch.optim.Adam(player.parameters(), lr=1e-3)
    batch_sampler = BatchSamplerGPUStorage(samples)
    for _ in range(mini_batches):
        obses, ts, values = batch_sampler(8192)
        optimizer.zero_grad()
        value_per_action = player(obses)
        loss = (ts * (value_per_action - values).pow(2)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(player.parameters(), max_norm=1.0)
        optimizer.step()


def train_policy(policy, policy_storage, logger):
    batch_sampler = BatchSamplerGPUStorage(policy_storage)

    epochs = 500
    batch_size = 8192
    mini_batches = epochs * len(policy_storage) // batch_size
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for iter in tqdm(range(mini_batches)):
        obses, ts, distributions = batch_sampler(batch_size)
        optimizer.zero_grad()
        action_distribution = policy(obses)
        action_distribution = torch.nn.functional.softmax(action_distribution, dim=-1)
        loss = (ts * (action_distribution - distributions).pow(2)).mean()
        logger.add_scalar("policy_training/loss", loss.item(), iter)
        loss.backward()
        optimizer.step()


def regret_matching(values, eps=1e-6):
    values = torch.clamp(values, min=0)
    total = torch.sum(values)
    if total <= eps:
        return torch.ones_like(values) / values.shape[-1]

    values /= torch.sum(values) + eps
    return values


class Workers:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.mp_pool = mp.Pool(num_processes)

    def __del__(self):
        self.mp_pool.close()

    def sync_map(self, func, args):
        return [func(arg) for arg in args]

    def async_map(self, func, args):
        return self.mp_pool.map(func, args)


def traverse_cfr(env, player_idx, players, samples_storage, policy_storage, cfr_iter):
    if env.done:
        return env.reward[player_idx]

    obs = env.obs
    batched_obs = _batch_obses([obs])
    if player_idx == obs["player_idx"]:
        values = players[player_idx](batched_obs)[0]
        distribution = regret_matching(values).cpu()
        va = torch.zeros(len(Action))
        mean_value_action = 0
        for action_idx, (probability, action) in enumerate(zip(distribution, Action)):
            cfr_env = deepcopy(env)
            cfr_env.step(action)

            value_per_action = traverse_cfr(
                cfr_env, player_idx, players, samples_storage, policy_storage, cfr_iter
            )
            va[action_idx] = value_per_action
            mean_value_action += probability * value_per_action
        va -= mean_value_action
        samples_storage[player_idx].append((obs, cfr_iter, va.numpy()))
        return mean_value_action
    else:
        values = players[1 - player_idx](batched_obs)[0]
        distribution = regret_matching(values).cpu()
        policy_storage.append((obs, cfr_iter, distribution.numpy()))
        sampled_action = torch.multinomial(distribution, 1).item()
        env.step(sampled_action)
        return traverse_cfr(
            env, player_idx, players, samples_storage, policy_storage, cfr_iter
        )


class TraversalWorker:
    def __init__(self, env, player_idx, players, cfr_iter):
        self.player_idx = player_idx
        self.cfr_iter = cfr_iter
        self.env = CFREnvWrapper(env)
        for idx in range(2):
            torch.save(players[idx].state_dict(), f"/tmp/player_{idx}.pth")

    def __call__(self, traverses):
        value_storage = [[], []]
        policy_storage = []

        players = [BaseModel() for _ in range(2)]
        for idx in range(2):
            players[idx].load_state_dict(
                torch.load(
                    f"/tmp/player_{idx}.pth", weights_only=True, map_location="cpu"
                )
            )
            players[idx].eval()

        with torch.no_grad():
            for _ in range(traverses):
                self.env.reset()
                traverse_cfr(
                    self.env,
                    self.player_idx,
                    players,
                    value_storage,
                    policy_storage,
                    self.cfr_iter,
                )
        return value_storage[self.player_idx], policy_storage


def deepcfr(env, cfr_iterations, traverses_per_iteration):
    num_players = 2
    assert num_players == 2
    players = [BaseModel().cuda() for _ in range(num_players)]
    policy = BaseModel().cuda()

    samples_storage = [
        GPUBoundedStorage(BOUNDED_STORAGE_MAX_SIZE) for _ in range(num_players)
    ]
    policy_storage = GPUBoundedStorage(BOUNDED_STORAGE_MAX_SIZE)

    timers = Timers()
    global_iter = 0

    logger = SummaryWriter()
    workers = Workers(NUM_WORKERS)
    for cfr_iter in tqdm(range(cfr_iterations)):
        for player_idx in range(num_players):
            timers.start("traverse")
            traversal_worker = TraversalWorker(
                env,
                player_idx,
                players,
                cfr_iter + 1,
            )

            per_worker_traverses = (
                traverses_per_iteration + workers.num_processes - 1
            ) // workers.num_processes

            results = workers.async_map(
                traversal_worker,
                [per_worker_traverses] * workers.num_processes,
            )
            traverse_time = timers.stop("traverse")
            logger.add_scalar("traverse_time", traverse_time, global_iter)

            timers.start("store traverse results")
            for value, pol in results:
                samples_storage[player_idx].add_all(value)
                policy_storage.add_all(pol)
            store_traverse_results_time = timers.stop("store traverse results")
            logger.add_scalar(
                "store_traverse_results_time", store_traverse_results_time, global_iter
            )

            total_values_to_add = sum(len(value) for value, _ in results)
            total_policy_to_add = sum(len(pol) for _, pol in results)
            logger.add_scalar("total_values_to_add", total_values_to_add, global_iter)
            logger.add_scalar("total_policy_to_add", total_policy_to_add, global_iter)

            players[player_idx] = BaseModel().cuda()
            timers.start("train values model")
            train_values(players[player_idx], samples_storage[player_idx])
            train_values_model_time = timers.stop("train values model")
            logger.add_scalar(
                "train_values_model_time", train_values_model_time, global_iter
            )
            global_iter += 1

            logger.add_scalar(
                "memory_usage", psutil.virtual_memory().percent, global_iter
            )
            logger.add_scalar(
                f"{player_idx}_samples_storage_size",
                samples_storage[player_idx].current_len,
                global_iter,
            )

    print("Policy storage size:", len(policy_storage))
    timers.start("train policy")
    train_policy(policy, policy_storage, logger)
    train_policy_time = timers.stop("train policy")
    logger.add_scalar("train_policy_time", train_policy_time, 0)
    torch.save(policy.state_dict(), "policy.pth")

    eval_policy_player_helper = EvalPolicy(env, policy, logger)
    print("Final policy evaluation:")
    eval_games_to_play = 50000
    timers.start("eval")
    eval_policy_player_helper.eval(games_to_play=eval_games_to_play)
    eval_time = timers.stop("eval")
    logger.add_scalar("eval_time", eval_time, 0)
    print("Deep CFR training complete")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    env = HeadsUpPoker(ObsProcessor())

    cfr_iterations = 200
    traverses_per_iteration = 3000
    deepcfr(env, cfr_iterations, traverses_per_iteration)
