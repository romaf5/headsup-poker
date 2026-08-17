"""DeepCFR advantage / policy network, in switchable variants.

The network consumes the flat float32 observation produced by
:meth:`headsup.engine.HeadsUpPoker.observation` (the first ``obs_dim`` entries of it) and
outputs one value per action.  Variants (``BaseModel(**config)``):

* ``features``: which bet features feed the bet branch.  ``aggregated`` = the 8 pot-normalised
  totals of the original design (obs[23:31]); ``history`` = the DeepCFR paper's per-street bet
  history (obs[31:79]: HISTORY_ROUNDS x HISTORY_SLOTS x [size / pot, occurred]) plus stack / pot
  and pot / 1000; ``both`` = all of them.
* ``arch``: ``current`` = card branch + stage/position embedding branch + bet branch (3 x dim
  into the trunk); ``paper`` = Figure 1 of Brown et al. (2019): card branch + bet branch only
  (2 x dim into the trunk), one card embedding per card group (hole, flop, turn, river), the
  position flag appended to the bet features.
* ``cards``: ``embed`` = rank + suit + card embeddings summed per card group (DeepCFR);
  ``onehot`` = concatenated one-hot cards straight into the first card layer (SD-CFR paper).
* ``dim``: width of every hidden layer and embedding (64 = ~68k / 66k parameters).
* ``rm_fallback``: what regret matching plays when no advantage is positive - ``uniform`` (over
  the allowed actions) or ``argmax`` (the highest advantage, DeepCFR paper).  Stored with the
  advantage nets because the average strategy is defined through it.
* ``game``: the action tree the network was trained for (:meth:`headsup.game.GameConfig.tree_dict`:
  ``bet_sizes``, ``raise_cap``, ``mask_redundant``); the head has ``len(bet_sizes) + 3`` outputs.

Models are saved as ``{"config": ..., "state_dict": ...}`` (:meth:`BaseModel.save`); the config
travels with every artefact (policy.pth, iterates.pt, checkpoints, numpy weight dicts).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from headsup.engine import HISTORY_DIM, HISTORY_OFFSET, OBS_DIM, OBS_DIM_AGGREGATED, OBS_DIM_HISTORY
from headsup.game import DEFAULT_GAME, GameConfig

SUITS = 4
RANKS = 13
NUM_STAGES = 4
CARD_SLOTS = 7  # hole 2, flop 3, turn 1, river 1
CARD_CLASSES = RANKS * SUITS + 1  # 0 = no card
OPP_CARDS_OFFSET = OBS_DIM  # history inputs: the opponent's two hole cards (rank+1, suit+1, card+1 x 2) after the observation
OBS_DIM_WITH_OPP = OBS_DIM + 6
EMBEDDING_DIM = 64  # default width

FEATURES = ("aggregated", "history", "both")
ARCHS = ("current", "paper")
CARDS = ("embed", "onehot")
RM_FALLBACKS = ("uniform", "argmax")
DEFAULT_CONFIG = dict(features="aggregated", arch="current", cards="embed", dim=EMBEDDING_DIM, rm_fallback="uniform",
                      game=DEFAULT_GAME.tree_dict(), opp_cards=False)


def normalize_config(config=None, **overrides):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in (config or {}).items() if v is not None})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    game = cfg["game"]
    cfg["game"] = (game if isinstance(game, GameConfig) else GameConfig.from_dict(game)).tree_dict()
    if cfg["features"] not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}, got {cfg['features']!r}")
    if cfg["arch"] not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {cfg['arch']!r}")
    if cfg["cards"] not in CARDS:
        raise ValueError(f"cards must be one of {CARDS}, got {cfg['cards']!r}")
    if cfg["rm_fallback"] not in RM_FALLBACKS:
        raise ValueError(f"rm_fallback must be one of {RM_FALLBACKS}, got {cfg['rm_fallback']!r}")
    cfg["dim"] = int(cfg["dim"])
    cfg["opp_cards"] = bool(cfg["opp_cards"])
    return cfg


def obs_dim_for(features, opp_cards=False):
    """Observation width a network with these bet features reads (a prefix of the full layout);
    history-input networks (``opp_cards``: the DREAM baseline / ESCHER value nets, which see both
    players' hole cards) read the full observation plus the opponent's cards appended to it."""
    if opp_cards:
        return OBS_DIM_WITH_OPP
    return OBS_DIM_AGGREGATED if features == "aggregated" else OBS_DIM_HISTORY


def history_observation(obs, opp_cards):
    """(B, OBS_DIM) observation rows of a seat + that seat's opponent's hole cards (B, 2) ->
    (B, OBS_DIM_WITH_OPP) history inputs (rank+1, suit+1, card+1 of the sorted opponent cards)."""
    from headsup.cards import CARD_FEATURES

    obs = np.asarray(obs, dtype=np.float32)
    opp = np.sort(np.asarray(opp_cards, dtype=np.int64).reshape(-1, 2), axis=1)
    feats = CARD_FEATURES[opp].reshape(len(opp), 6).astype(np.float32)
    return np.concatenate([obs, feats], axis=1)


def bet_feature_indices(features, arch):
    """Observation indices fed to the bet branch (mirrored by the numpy and C++ forward passes)."""
    idx = []
    if features in ("aggregated", "both"):
        idx += list(range(23, OBS_DIM_AGGREGATED))
    if features in ("history", "both"):
        idx += list(range(HISTORY_OFFSET, HISTORY_OFFSET + HISTORY_DIM))
        if features == "history":
            idx += [28, 29]  # stack / pot and pot / 1000 (the SD-CFR paper adds "the size of the pot")
    if arch == "paper":
        idx += [22]  # position: the paper net has no stage/position branch
    return idx


class CardEmbedding(nn.Module):
    """rank + suit + card embeddings of every card slot; index 0 = no card (padding row)."""

    def __init__(self, dim):
        super().__init__()
        self.rank_embedding = nn.Embedding(RANKS + 1, dim)
        self.suit_embedding = nn.Embedding(SUITS + 1, dim)
        self.card_embedding = nn.Embedding(RANKS * SUITS + 1, dim)

    def forward(self, cards):  # cards: (B, n, 3) long
        return self.rank_embedding(cards[..., 0]) + self.suit_embedding(cards[..., 1]) + self.card_embedding(cards[..., 2])


def _group_sums(emb):
    """(B, 7, dim) per-slot embeddings -> (B, 4 * dim): hole, flop, turn, river sums.

    Elementwise adds instead of .sum(dim=1): identical result, but ~30x faster on MPS, whose
    reduction kernel over a short strided dim is pathologically slow.
    """
    hand = emb[:, 0] + emb[:, 1]
    flop = emb[:, 2] + emb[:, 3] + emb[:, 4]
    return torch.cat([hand, flop, emb[:, 5], emb[:, 6]], dim=1)


class CardModel(nn.Module):
    """Card branch: 3 layers; input = summed embeddings per group or concatenated one-hot cards."""

    def __init__(self, dim=EMBEDDING_DIM, cards="embed", per_group=False, opp_cards=False):
        super().__init__()
        self.cards = cards
        self.per_group = per_group
        self.opp_cards = opp_cards
        n_groups, n_slots = (5, 9) if opp_cards else (4, 7)
        if cards == "embed":
            if per_group:  # DeepCFR paper: one embedding per card group
                self.group_embeddings = nn.ModuleList([CardEmbedding(dim) for _ in range(n_groups)])
            else:
                self.cards_embeddings = CardEmbedding(dim)
            self.fc1 = nn.Linear(n_groups * dim, dim)
        else:  # SD-CFR paper: one-hot cards, so the first layer is the (only) embedding
            self.onehot = nn.Linear(n_slots * CARD_CLASSES, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, cards):  # (B, 7 or 9, 3) long
        if self.cards == "embed":
            if self.per_group:
                g = self.group_embeddings
                groups = [
                    g[0](cards[:, 0]) + g[0](cards[:, 1]),
                    g[1](cards[:, 2]) + g[1](cards[:, 3]) + g[1](cards[:, 4]),
                    g[2](cards[:, 5]),
                    g[3](cards[:, 6]),
                ]
                if self.opp_cards:
                    groups.append(g[4](cards[:, 7]) + g[4](cards[:, 8]))
                x = torch.cat(groups, dim=1)
            else:
                emb = self.cards_embeddings(cards)
                x = _group_sums(emb)
                if self.opp_cards:
                    x = torch.cat([x, emb[:, 7] + emb[:, 8]], dim=1)
            x = self.act(self.fc1(x))
        else:
            onehot = F.one_hot(cards[:, :, 2], CARD_CLASSES).to(self.onehot.weight.dtype).flatten(1)
            x = self.act(self.onehot(onehot))
        x = self.act(self.fc2(x))
        return self.act(self.fc3(x))


class StageAndOrderModel(nn.Module):
    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        self.stage_embedding = nn.Embedding(NUM_STAGES, dim)
        self.first_to_act_embedding = nn.Embedding(2, dim)
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, stage, first_to_act):
        x = torch.cat([self.stage_embedding(stage), self.first_to_act_embedding(first_to_act)], dim=1)
        x = self.act(self.fc1(x))
        return self.act(self.fc2(x))


class BetsModel(nn.Module):
    def __init__(self, n_features=8, dim=EMBEDDING_DIM):
        super().__init__()
        self.fc1 = nn.Linear(n_features, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.act(self.fc2(x) + x)


class BaseModel(nn.Module):
    """Maps a batch of observations (B, >= obs_dim) to per-action logits / advantages (B, num_actions)."""

    def __init__(self, features=None, arch=None, cards=None, dim=None, rm_fallback=None, game=None, config=None, opp_cards=None):
        super().__init__()
        cfg = normalize_config(config, features=features, arch=arch, cards=cards, dim=dim, rm_fallback=rm_fallback, game=game,
                               opp_cards=opp_cards)
        self.config = cfg
        self.features, self.arch, self.cards_mode = cfg["features"], cfg["arch"], cfg["cards"]
        self.dim, self.rm_fallback = cfg["dim"], cfg["rm_fallback"]
        self.opp_cards = cfg["opp_cards"]
        self.game = GameConfig.from_dict(cfg["game"])  # action tree (stack / blinds at their defaults)
        self.obs_dim = obs_dim_for(self.features, self.opp_cards)
        self.num_actions = self.game.num_actions
        d = self.dim
        idx = bet_feature_indices(self.features, self.arch)
        self.register_buffer("bet_index", torch.tensor(idx, dtype=torch.long), persistent=False)
        # upper bounds of the integer features (rank+1, suit+1, card+1 per card slot); indices are
        # clamped in forward() so that a corrupted sample in a 20M-row GPU memory (seen once: a
        # single flipped bit on a non-ECC card) cannot trigger a device-side assert and kill a run
        self.register_buffer("card_max", torch.tensor([RANKS, SUITS, RANKS * SUITS], dtype=torch.long), persistent=False)

        self.card_model = CardModel(d, self.cards_mode, per_group=self.arch == "paper", opp_cards=self.opp_cards)
        if self.arch == "current":
            self.stage_and_order_model = StageAndOrderModel(d)
        self.bets_model = BetsModel(len(idx), d)

        self.act = nn.ReLU()
        self.comb1 = nn.Linear((3 if self.arch == "current" else 2) * d, d)
        self.comb2 = nn.Linear(d, d)
        self.comb3 = nn.Linear(d, d)
        self.action_head = nn.Linear(d, self.num_actions)
        # zero head -> uniform initial strategy
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    @staticmethod
    def normalize(z):
        return (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)

    def forward(self, obs):
        if obs.shape[1] != self.obs_dim:
            if obs.shape[1] < self.obs_dim:
                raise ValueError(f"observation has {obs.shape[1]} features, this network needs {self.obs_dim}")
            obs = obs[:, : self.obs_dim]
        cards = obs[:, :21].long().view(-1, 7, 3)
        if self.opp_cards:
            cards = torch.cat([cards, obs[:, OPP_CARDS_OFFSET:OPP_CARDS_OFFSET + 6].long().view(-1, 2, 3)], dim=1)
        cards = torch.minimum(cards, self.card_max).clamp_(min=0)
        bets = obs.index_select(1, self.bet_index)
        parts = [self.card_model(cards)]
        if self.arch == "current":
            parts.append(self.stage_and_order_model(obs[:, 21].long().clamp(0, NUM_STAGES - 1), obs[:, 22].long().clamp(0, 1)))
        parts.append(self.bets_model(bets))
        z = torch.cat(parts, dim=1)
        z = self.act(self.comb1(z))
        z = self.act(self.comb2(z) + z)
        z = self.act(self.comb3(z) + z)
        z = self.normalize(z)
        return self.action_head(z)

    # -- (de)serialisation ------------------------------------------------------------
    def state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def numpy_weights(self):
        """State dict as float32 numpy arrays plus ``"config"`` (for the numpy / C++ mirrors)."""
        w = {k: v.detach().cpu().float().numpy() for k, v in self.state_dict().items()}
        w["config"] = dict(self.config)
        return w

    def save(self, path):
        torch.save({"config": dict(self.config), "state_dict": self.state_dict_cpu()}, path)

    @staticmethod
    def from_state(state, strict=True):
        """Build a model from ``{"config", "state_dict"}`` (the :meth:`save` format)."""
        model = BaseModel(config=state["config"])
        model.load_state_dict(state["state_dict"], strict=strict)
        return model


def load_model(path, device=None, strict=True) -> BaseModel:
    """Load a model written by :meth:`BaseModel.save`."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = BaseModel.from_state(state, strict=strict)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
