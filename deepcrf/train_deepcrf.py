import time
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from model import BaseModel
from bounded_storage import BoundedStorage
from player_wrapper import PolicyPlayerWrapper
from pokerenv_crf import Action, HeadsUpPoker, ObsProcessor
from simple_players import RandomPlayer, AlwaysCallPlayer, AlwaysAllInPlayer

NUM_WORKERS = 16
BOUNDED_STORAGE_MAX_SIZE = 40_000_000


class ValuePlayerWrapper:
    def __init__(self, player):
        self.player = player

    def __call__(self, obs):
        with torch.no_grad():
            obs = _batch_obses([obs])
            values = self.player(obs)[0]
            action = torch.multinomial(regret_matching(values), 1).item()
            return action


class EvalValuePlayers:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger
        self.opponent_players = {
            "random": RandomPlayer(),
            "call": AlwaysCallPlayer(),
            "allin": AlwaysAllInPlayer(),
        }

    def eval(self, players, crf_iter, games_to_play=1000):
        for opponent_name, opponent_player in self.opponent_players.items():
            for i, player in enumerate(players):
                wrapped_player = ValuePlayerWrapper(player)
                rewards = []
                for _ in range(games_to_play):
                    obs = self.env.reset()
                    done = False
                    while not done:
                        if obs["player_idx"] == i:
                            action = wrapped_player(obs)
                        else:
                            action = opponent_player(obs)
                        obs, reward, done, _ = self.env.step(action)
                        if done:
                            rewards.append(reward[i])
                self.logger.add_scalar(
                    f"value_network/{opponent_name}/player_{i}/mean_reward",
                    np.mean(rewards),
                    crf_iter,
                )


class EvalPolicyPlayer:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger
        self.opponent_players = {
            "random": RandomPlayer(),
            "call": AlwaysCallPlayer(),
            "allin": AlwaysAllInPlayer(),
        }

    def eval(self, player, games_to_play=1000):
        for opponent_name, opponent_player in self.opponent_players.items():
            wrapped_player = PolicyPlayerWrapper(player)
            rewards = []
            for play_as_idx in [0, 1]:
                for _ in range(games_to_play):
                    obs = self.env.reset()
                    done = False
                    while not done:
                        if obs["player_idx"] == play_as_idx:
                            action = wrapped_player(obs)
                        else:
                            action = opponent_player(obs)
                        obs, reward, done, _ = self.env.step(action)
                        if done:
                            rewards.append(reward[play_as_idx])
            self.logger.add_scalar(
                f"policy/{opponent_name}/mean_reward", np.mean(rewards), 0
            )


def _batch_obses(obses):
    return {k: torch.tensor([obs[k] for obs in obses]).cuda() for k in obses[0].keys()}


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


def train_values(player, samples):
    mini_batches = 4000
    optimizer = torch.optim.Adam(player.parameters(), lr=1e-3)
    batch_sampler = BatchSampler(samples)
    for _ in range(mini_batches):
        obses, ts, values = batch_sampler(8192)
        optimizer.zero_grad()
        value_per_action = player(obses)
        loss = (ts * (value_per_action - values).pow(2)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(player.parameters(), max_norm=1.0)
        optimizer.step()


def train_policy(policy, policy_storage):
    batch_sampler = BatchSampler(policy_storage)

    epochs = 100
    batch_size = 8192
    mini_batches = epochs * len(policy_storage) // batch_size
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(mini_batches):
        obses, ts, distributions = batch_sampler(batch_size)
        optimizer.zero_grad()
        action_distribution = policy(obses)
        action_distribution = torch.nn.functional.softmax(action_distribution, dim=-1)
        loss = (ts * (action_distribution - distributions).pow(2)).mean()
        loss.backward()
        optimizer.step()


class CRFEnvWrapper:
    def __init__(self, env):
        self.env = deepcopy(env)
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        self.reward = None
        self.done = False
        self.info = None
        return self.obs

    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        return self.obs, self.reward, self.done, self.info


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

    def map(self, func, args):
        return self.mp_pool.map(func, args)


def traverse_crf(env, player_idx, players, samples_storage, policy_storage, crf_iter):
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
            crf_env = deepcopy(env)
            crf_env.step(action)

            value_per_action = traverse_crf(
                crf_env, player_idx, players, samples_storage, policy_storage, crf_iter
            )
            va[action_idx] = value_per_action
            mean_value_action += probability * value_per_action
        va -= mean_value_action
        samples_storage[player_idx].append((obs, crf_iter, va.numpy()))
        return mean_value_action
    else:
        values = players[1 - player_idx](batched_obs)[0]
        distribution = regret_matching(values).cpu()
        policy_storage.append((obs, crf_iter, distribution.numpy()))
        sampled_action = torch.multinomial(distribution, 1).item()
        env.step(sampled_action)
        return traverse_crf(
            env, player_idx, players, samples_storage, policy_storage, crf_iter
        )


class TraversalWorker:
    def __init__(self, env, player_idx, players, crf_iter):
        self.player_idx = player_idx
        self.players = players
        self.crf_iter = crf_iter
        self.env = CRFEnvWrapper(env)

    def __call__(self, traverses):
        value_storage = [[], []]
        policy_storage = []
        with torch.no_grad():
            for _ in range(traverses):
                self.env.reset()
                traverse_crf(
                    self.env,
                    self.player_idx,
                    self.players,
                    value_storage,
                    policy_storage,
                    self.crf_iter,
                )
        return value_storage[self.player_idx], policy_storage


class Timers:
    def __init__(self):
        self.timers = {}

    def start(self, name):
        self.timers[name] = time.time()

    def stop(self, name):
        self.timers[name] = time.time() - self.timers[name]
        return self.timers[name]


def deepcrf(env, crf_iterations, traverses_per_iteration):
    num_players = 2
    assert num_players == 2
    players = [BaseModel().cuda() for _ in range(num_players)]
    policy = BaseModel().cuda()

    samples_storage = [
        BoundedStorage(BOUNDED_STORAGE_MAX_SIZE) for _ in range(num_players)
    ]
    policy_storage = []

    logger = SummaryWriter()
    eval_value_player_helper = EvalValuePlayers(env, logger)
    workers = Workers(NUM_WORKERS)
    timers = Timers()
    for crf_iter in tqdm(range(crf_iterations)):
        for player_idx in range(num_players):
            timers.start("traversal")
            traversal_worker = TraversalWorker(
                env,
                player_idx,
                players,
                crf_iter + 1,
            )

            per_worker_traverses = (
                traverses_per_iteration + workers.num_processes - 1
            ) // workers.num_processes

            results = workers.map(
                traversal_worker,
                [per_worker_traverses] * workers.num_processes,
            )
            for value, pol in results:
                samples_storage[player_idx].add_all(value)
                policy_storage.extend(pol)
            print(
                f"Cfr iteration {crf_iter} Player {player_idx} traversed {timers.stop('traversal'):.2f} seconds"
            )
            players[player_idx] = BaseModel().cuda()
            timers.start("train values")
            train_values(players[player_idx], samples_storage[player_idx].get_storage())
            print("Train values time:", timers.stop("train values"))
        # evaluate against random, call, allin players
        eval_value_player_helper.eval(players, crf_iter)

    print("Policy storage size:", len(policy_storage))
    train_policy(policy, policy_storage)
    eval_policy_player_helper = EvalPolicyPlayer(env, logger)
    print("\nFinal policy evaluation:")
    eval_games_to_play = 50000
    timers.start("eval policy")
    eval_policy_player_helper.eval(policy, games_to_play=eval_games_to_play)
    print("Eval policy time:", timers.stop("eval policy"))
    print("Deep CRF training complete")
    torch.save(policy.state_dict(), "policy.pth")
    print("Policy model saved!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
    env = HeadsUpPoker(ObsProcessor())

    crf_iterations = 1024
    traverses_per_iteration = 16384
    deepcrf(env, crf_iterations, traverses_per_iteration)
