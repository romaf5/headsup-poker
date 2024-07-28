import numpy as np

import torch
from treys import Card
from poker_env import ObsProcessor
from poker_env import HeadsUpPoker, Action
from poker_env import RandomPlayer, AlwaysCallPlayer


RANKS = 13
SUITS = 4
EMBEDDING_DIM = 128


class SimpleNetwork(torch.nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.num_actions = len(Action)
        self.rank_embedding = torch.nn.Embedding(RANKS + 1, EMBEDDING_DIM)
        self.suit_embedding = torch.nn.Embedding(SUITS + 1, EMBEDDING_DIM)
        self.card_embedding = torch.nn.Embedding(RANKS * SUITS + 1, EMBEDDING_DIM)
        self.stage_embedding = torch.nn.Embedding(4, EMBEDDING_DIM)
        self.first_to_act_embedding = torch.nn.Embedding(2, EMBEDDING_DIM)
        self.act = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear((15 + 6 + 1 + 1) * EMBEDDING_DIM + 8, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, self.num_actions)

    def forward(self, x):
        stage = x["stage"]
        board_and_hand = x["board_and_hand"]
        first_to_act_next_stage = x["first_to_act_next_stage"].long()

        # B, 21
        batch_size = board_and_hand.size(0)
        board_and_hand = board_and_hand.view(batch_size, 7, 3)

        ranks = board_and_hand[:, :, 0].long()
        suits = board_and_hand[:, :, 1].long()
        card_indices = board_and_hand[:, :, 2].long()

        ranks_emb = self.rank_embedding(ranks)
        suits_emb = self.suit_embedding(suits)
        card_indices_emb = self.card_embedding(card_indices)

        board_and_hand_emb = torch.cat([ranks_emb, suits_emb, card_indices_emb], dim=2)
        board_and_hand_emb = board_and_hand_emb.view(batch_size, -1)

        stage_emb = self.stage_embedding(stage)
        first_to_act_next_stage_emb = self.first_to_act_embedding(
            first_to_act_next_stage
        )

        all_emb = torch.cat(
            [
                board_and_hand_emb,
                stage_emb,
                first_to_act_next_stage_emb,
                x["bets_and_stacks"],
            ],
            dim=1,
        )

        x = self.fc1(all_emb)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SimpleModel:
    def __init__(self):
        self.nn_model = SimpleNetwork()
        self.nn_model.cuda()

        self.loss_fn = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=1e-3)

    def save(self, path):
        torch.save(self.nn_model.state_dict(), path)

    def load(self, path):
        self.nn_model.load_state_dict(torch.load(path))
        self.nn_model.cuda()

    def _generate_samples(self, env, hands_to_play=128):
        obses = []
        actions = []
        rewards = []
        rewards_per_hand = []
        for _ in range(hands_to_play):
            obs = env.reset()
            done = False
            while not done:
                obses.append(obs.copy())
                action = self(obs)
                actions.append(action)
                obs, reward, done, _ = env.step(action)
                if done:
                    rewards_per_hand.append(reward)
                    rewards.extend([reward] * (len(obses) - len(rewards)))
        samples_info = {
            "mean_reward": np.mean(rewards_per_hand),
            "hands_played": hands_to_play,
        }
        return obses, actions, rewards, samples_info

    def _batch_obses(self, obses):
        batch_obses = {}
        for k in obses[0].keys():
            batch_obses[k] = []
        for obs in obses:
            for k, v in obs.items():
                batch_obses[k].append(v)
        for k in batch_obses.keys():
            batch_obses[k] = torch.tensor(batch_obses[k]).cuda()
        return batch_obses

    def _train_batch(self, obses, actions, rewards):
        obses = self._batch_obses(obses)
        actions = torch.tensor(actions).cuda().long()
        rewards = torch.tensor(rewards).cuda().float()

        self.optimizer.zero_grad()
        value_per_action = self.nn_model(obses)  # (B, Actions)
        value = value_per_action[torch.arange(len(actions)), actions]
        loss = self.loss_fn(value, rewards)
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, env):
        average_meter = AverageMeter()
        for _ in range(10):
            obses, actions, rewards, info = self._generate_samples(env)
            self._train_batch(obses, actions, rewards)
            average_meter.update(info["mean_reward"], info["hands_played"])
        return average_meter.avg

    def __call__(self, obs):
        assert type(obs) == dict

        for k, v in obs.items():
            obs[k] = torch.tensor(v).unsqueeze(0).cuda()

        with torch.no_grad():
            value_per_action = self.nn_model(obs)
        max_value_action = torch.max(value_per_action)
        if max_value_action < 0:
            return torch.argmax(value_per_action).item()
        else:
            values_per_action = value_per_action.squeeze().cpu().detach().numpy()
            values_per_action[values_per_action < 0] = 0
            values_per_action = values_per_action / values_per_action.sum()
            return np.random.choice(range(len(values_per_action)), p=values_per_action)


def main():
    model = SimpleModel()
    obs_processor = ObsProcessor()
    env = HeadsUpPoker(obs_processor, AlwaysCallPlayer())
    for epoch in range(100):
        avg_reward = model.train_epoch(env)
        print("Epoch:", epoch + 1, "Avg reward:", avg_reward)
    model.save("simple_model.pth")
    # evaluate
    model.load("simple_model.pth")
    _, _, _, info = model._generate_samples(env, hands_to_play=10000)
    print("Eval mean reward:", info["mean_reward"])


if __name__ == "__main__":
    main()
