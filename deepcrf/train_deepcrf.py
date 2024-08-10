from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from model import BaseModel
from player_wrapper import PolicyPlayerWrapper
from pokerenv_crf import Action, HeadsUpPoker, ObsProcessor

NUM_WORKERS = 16


class AlwaysCallPlayer:
    def __call__(self, _):
        return Action.CHECK_CALL


class AlwaysAllInPlayer:
    def __call__(self, _):
        return Action.ALL_IN


class RandomPlayer:
    def __call__(self, _):
        return np.random.choice(
            [Action.FOLD, Action.CHECK_CALL, Action.RAISE, Action.ALL_IN]
        )


class ValuePlayerWrapper:
    def __init__(self, player):
        self.player = player

    def __call__(self, obs):
        with torch.no_grad():
            obs = _batch_obses([obs])
            values = self.player(obs)
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
    batch_obses = {}
    for k in obses[0].keys():
        batch_obses[k] = []
    for obs in obses:
        for k, v in obs.items():
            batch_obses[k].append(v)
    for k in batch_obses.keys():
        batch_obses[k] = torch.tensor(batch_obses[k]).cuda()
    return batch_obses


class ValuesDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    obses = [sample[0] for sample in batch]
    obses = _batch_obses(obses)
    ts = torch.tensor([sample[1] for sample in batch]).cuda()[..., None]
    values = torch.cat([sample[2][None] for sample in batch]).cuda()
    return obses, ts, values


def train_values(player, samples):
    dataset = ValuesDataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True,
        collate_fn=_collate_fn,
    )

    mini_batches = 4000
    batches_per_loader = len(loader)
    runs = (mini_batches + batches_per_loader - 1) // batches_per_loader
    optimizer = torch.optim.Adam(player.parameters(), lr=1e-3)
    for _ in range(runs):
        for obses, ts, values in loader:
            optimizer.zero_grad()
            value_per_action = player(obses)
            loss = ((value_per_action - values).pow(2)).mean()
            loss.backward()
            optimizer.step()


class CRFEnvWrapper:
    def __init__(self, env):
        self.env = env
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
        return torch.ones_like(values) / len(values)

    values /= torch.sum(values) + eps
    return values


class Workers:
    def __init__(self, num_processes=NUM_WORKERS):
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
        values = players[player_idx](batched_obs)
        distribution = regret_matching(values).cpu()
        va = torch.zeros(len(Action))
        mean_value_action = 0
        for probability, action in zip(distribution, Action):
            crf_env = deepcopy(env)
            crf_env.step(action)

            value_per_action = traverse_crf(
                crf_env, player_idx, players, samples_storage, policy_storage, crf_iter
            )
            mean_value_action += probability * value_per_action
        va -= mean_value_action
        samples_storage[player_idx].append((obs, crf_iter, va))
    else:
        values = players[1 - player_idx](batched_obs)
        distribution = regret_matching(values).cpu()
        policy_storage.append((obs, crf_iter, distribution))
        sampled_action = torch.multinomial(distribution, 1).item()
        env.step(sampled_action)
        return traverse_crf(
            env, player_idx, players, samples_storage, policy_storage, crf_iter
        )


def train_policy(policy, policy_storage):
    dataset = ValuesDataset(policy_storage)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True,
        collate_fn=_collate_fn,
    )

    epochs = 100
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(epochs):
        for obses, ts, values in loader:
            optimizer.zero_grad()
            action_distribution = policy(obses)
            loss = ((action_distribution - values).pow(2)).mean()
            loss.backward()
            optimizer.step()


class TraversalWorker:
    def __init__(self, env, player_idx, players, crf_iter):
        self.player_idx = player_idx
        self.players = players
        self.value_storage = [[], []]
        self.policy_storage = []
        self.crf_iter = crf_iter
        self.env = CRFEnvWrapper(deepcopy(env))

    def __call__(self, traverses):
        with torch.no_grad():
            for _ in range(traverses):
                self.env.reset()
                traverse_crf(
                    self.env,
                    self.player_idx,
                    self.players,
                    self.value_storage,
                    self.policy_storage,
                    self.crf_iter,
                )
        return self.value_storage[self.player_idx], self.policy_storage


def deepcrf(env, crf_iterations, traverses_per_iteration):
    num_players = 2
    assert num_players == 2
    players = [BaseModel().cuda() for _ in range(num_players)]
    policy = BaseModel().cuda()

    samples_storage = [[] for _ in range(num_players)]
    policy_storage = []

    logger = SummaryWriter()
    eval_value_player_helper = EvalValuePlayers(env, logger)
    workers = Workers()

    for crf_iter in tqdm(range(crf_iterations)):
        for player_idx in range(num_players):
            traversal_worker = TraversalWorker(
                env,
                player_idx,
                players,
                crf_iter,
            )
            per_worker_traverses = (
                traverses_per_iteration + workers.num_processes - 1
            ) // workers.num_processes

            results = workers.map(
                traversal_worker, [per_worker_traverses] * workers.num_processes
            )

            for value_storage, policy_storage in results:
                samples_storage[player_idx].extend(value_storage)
                policy_storage.extend(policy_storage)

            players[player_idx] = BaseModel().cuda()
            train_values(players[player_idx], samples_storage[player_idx])
            # clean up storage
            samples_storage[player_idx] = []
        # evaluate against random, call, allin players
        eval_value_player_helper.eval(players, crf_iter)

    train_policy(policy, policy_storage)
    eval_policy_player_helper = EvalPolicyPlayer(env, logger)
    print("\nFinal policy evaluation:")
    eval_policy_player_helper.eval(policy)
    print("Deep CRF training complete")
    torch.save(policy.state_dict(), "policy.pth")
    print("Policy model saved!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    env = HeadsUpPoker(ObsProcessor())

    crf_iterations = 100
    traverses_per_iteration = 1000
    deepcrf(env, crf_iterations, traverses_per_iteration)
