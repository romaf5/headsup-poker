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
from pokerenv_cfr import Action, HeadsUpPoker, ObsProcessor
from simple_players import RandomPlayer, AlwaysCallPlayer, AlwaysAllInPlayer
from cfr_env_wrapper import CFREnvWrapper
from eval_policy import EvalPolicyPlayer, EvalExploitabilityScore

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

    def eval(self, players, cfr_iter, games_to_play=1000):
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
                    cfr_iter,
                )


class EvalPolicy:
    def __init__(self, env, policy, logger):
        self.env = env
        self.player = PolicyPlayerWrapper(policy)
        self.logger = logger

    def eval(self, games_to_play=50000):
        eval_policy_player = EvalPolicyPlayer(self.env)
        simple_player_scores = eval_policy_player.eval(self.player, games_to_play)

        eval_exploitability_score = EvalExploitabilityScore(self.env, self.player)
        exploitability_score_mbb_g = (
            eval_exploitability_score.eval(games_to_play) * 1000 / 2
        )

        for opponent_name, score in simple_player_scores.items():
            self.logger.add_scalar(
                f"policy_evaluation/{opponent_name}/mean_reward", score
            )
        self.logger.add_scalar(
            "policy_evaluation/exploitability_score_mbb_g", exploitability_score_mbb_g
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

        players = [BaseModel().cuda() for _ in range(2)]
        for idx in range(2):
            players[idx].load_state_dict(
                torch.load(f"/tmp/player_{idx}.pth", weights_only=True)
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


class Timers:
    def __init__(self):
        self.timers = {}

    def start(self, name):
        self.timers[name] = time.time()

    def stop(self, name):
        self.timers[name] = time.time() - self.timers[name]
        return self.timers[name]


def deepcfr(env, cfr_iterations, traverses_per_iteration):
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
    for cfr_iter in tqdm(range(cfr_iterations)):
        for player_idx in range(num_players):
            timers.start("traversal")
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
            for value, pol in results:
                samples_storage[player_idx].add_all(value)
                policy_storage.extend(pol)
            print(
                f"Cfr iteration {cfr_iter} Player {player_idx} traversed {timers.stop('traversal'):.2f} seconds"
            )
            players[player_idx] = BaseModel().cuda()
            timers.start("train values")
            train_values(players[player_idx], samples_storage[player_idx].get_storage())
            print("Train values time:", timers.stop("train values"))
        # evaluate against random, call, allin players
        eval_value_player_helper.eval(players, cfr_iter)

    print("Policy storage size:", len(policy_storage))
    train_policy(policy, policy_storage)
    torch.save(policy.state_dict(), "policy.pth")

    eval_policy_player_helper = EvalPolicy(env, policy, logger)
    print("\nFinal policy evaluation:")
    eval_games_to_play = 50000
    timers.start("eval policy")
    eval_policy_player_helper.eval(games_to_play=eval_games_to_play)
    print("Eval policy time:", timers.stop("eval policy"))
    print("Deep CFR training complete")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
    env = HeadsUpPoker(ObsProcessor())

    cfr_iterations = 256
    traverses_per_iteration = 10000
    deepcfr(env, cfr_iterations, traverses_per_iteration)
