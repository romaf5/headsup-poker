"""DeepCFR advantage / policy network.

The network consumes the flat float32[31] observation produced by
:meth:`headsup.engine.HeadsUpPoker.observation`.  Parameter names are unchanged from the
original dict-input model, so previously trained checkpoints load without conversion.
"""

import torch
import torch.nn as nn

from headsup.enums import NUM_ACTIONS

SUITS = 4
RANKS = 13
EMBEDDING_DIM = 64
NUM_STAGES = 4


class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rank_embedding = nn.Embedding(RANKS + 1, dim)
        self.suit_embedding = nn.Embedding(SUITS + 1, dim)
        self.card_embedding = nn.Embedding(RANKS * SUITS + 1, dim)

    def forward(self, cards):  # cards: (B, 7, 3) long
        emb = (
            self.rank_embedding(cards[:, :, 0])
            + self.suit_embedding(cards[:, :, 1])
            + self.card_embedding(cards[:, :, 2])
        )
        # elementwise adds instead of .sum(dim=1): identical result, but ~30x faster on MPS,
        # whose reduction kernel over a short strided dim is pathologically slow
        hand = emb[:, 0] + emb[:, 1]
        flop = emb[:, 2] + emb[:, 3] + emb[:, 4]
        turn = emb[:, 5]
        river = emb[:, 6]
        return torch.cat([hand, flop, turn, river], dim=1)


class CardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cards_embeddings = CardEmbedding(EMBEDDING_DIM)
        self.fc1 = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc3 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.act = nn.ReLU()

    def forward(self, cards):
        x = self.cards_embeddings(cards)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.act(self.fc3(x))


class StageAndOrderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_embedding = nn.Embedding(NUM_STAGES, EMBEDDING_DIM)
        self.first_to_act_embedding = nn.Embedding(2, EMBEDDING_DIM)
        self.fc1 = nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.act = nn.ReLU()

    def forward(self, stage, first_to_act):
        x = torch.cat(
            [self.stage_embedding(stage), self.first_to_act_embedding(first_to_act)], dim=1
        )
        x = self.act(self.fc1(x))
        return self.act(self.fc2(x))


class BetsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.act(self.fc2(x) + x)


class BaseModel(nn.Module):
    """Maps a batch of observations (B, 31) to per-action logits / advantages (B, 4)."""

    def __init__(self):
        super().__init__()
        self.num_actions = NUM_ACTIONS
        self.card_model = CardModel()
        self.stage_and_order_model = StageAndOrderModel()
        self.bets_model = BetsModel()

        self.act = nn.ReLU()
        self.comb1 = nn.Linear(3 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.comb2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.comb3 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.action_head = nn.Linear(EMBEDDING_DIM, NUM_ACTIONS)
        # zero head -> uniform initial strategy
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    @staticmethod
    def normalize(z):
        return (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)

    def forward(self, obs):
        cards = obs[:, :21].long().view(-1, 7, 3)
        stage = obs[:, 21].long()
        first_to_act = obs[:, 22].long()
        bets = obs[:, 23:31]

        z = torch.cat(
            [
                self.card_model(cards),
                self.stage_and_order_model(stage, first_to_act),
                self.bets_model(bets),
            ],
            dim=1,
        )
        z = self.act(self.comb1(z))
        z = self.act(self.comb2(z) + z)
        z = self.act(self.comb3(z) + z)
        z = self.normalize(z)
        return self.action_head(z)

    def numpy_weights(self):
        """State dict as float32 numpy arrays (for :class:`headsup.numpy_model.NumpyModel`)."""
        return {k: v.detach().cpu().float().numpy() for k, v in self.state_dict().items()}


def load_model(path, device=None, strict=True) -> BaseModel:
    """Load a checkpoint saved with ``torch.save(model.state_dict(), path)``."""
    model = BaseModel()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:  # trainer checkpoint format
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    model.eval()
    if device is not None:
        model.to(device)
    return model
