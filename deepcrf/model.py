import torch

SUITS = 4
RANKS = 13
EMBEDDING_DIM = 128

# Number of actions: fold, check/call, raise, all-in
NUM_ACTIONS = 4


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.num_actions = NUM_ACTIONS
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

        # Normalize the output
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        return self.fc3(x)
