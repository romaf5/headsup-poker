import torch
from copy import deepcopy
from tqdm import tqdm
from pokerenv_crf import Action, HeadsUpPoker, ObsProcessor

RANKS = 13
SUITS = 4
EMBEDDING_DIM = 128


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
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
        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data.fill_(0)

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


def deepcrf(env, crf_iterations, traverses_per_iteration):
    num_players = 2
    assert num_players == 2
    players = [BaseModel().cuda() for _ in range(num_players)]
    policy = BaseModel().cuda()

    samples_storage = [[] for _ in range(num_players)]
    policy_storage = []

    for crf_iter in tqdm(range(crf_iterations)):
        for player_idx in range(num_players):
            with torch.no_grad():
                for _ in range(traverses_per_iteration):
                    crf_env = CRFEnvWrapper(env)
                    traverse_crf(
                        crf_env,
                        player_idx,
                        players,
                        samples_storage,
                        policy_storage,
                        crf_iter,
                    )

            players[player_idx] = BaseModel().cuda()
            train_values(players[player_idx], samples_storage[player_idx])
            # clean up storage
            samples_storage[player_idx] = []
    train_policy(policy, policy_storage)
    print("Deep CRF training complete")


if __name__ == "__main__":
    env = HeadsUpPoker(ObsProcessor())

    crf_iterations = 1
    traverses_per_iteration = 1
    deepcrf(env, crf_iterations, traverses_per_iteration)
