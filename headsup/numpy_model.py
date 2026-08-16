"""Pure-numpy mirror of :class:`headsup.model.BaseModel` (all variants).

CFR traversals evaluate the network on one observation at a time from many CPU worker
processes.  Doing that in PyTorch costs ~0.4 ms per call in framework overhead; the same
arithmetic in numpy on 1-D vectors takes ~40 µs and the workers do not need to import
torch at all.  ``NumpyModel.__call__`` handles both a single observation (obs_dim,) and a
batch (B, >= obs_dim); wider observations are truncated to the network's ``obs_dim``.
"""

import numpy as np

from headsup.game import GameConfig
from headsup.model import CARD_CLASSES, bet_feature_indices, normalize_config, obs_dim_for


def _relu(x):
    return np.maximum(x, 0.0, out=x)


class NumpyModel:
    def __init__(self, weights: dict):
        weights = dict(weights)
        self.config = normalize_config(weights.pop("config"))
        w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.features, self.arch, self.cards = self.config["features"], self.config["arch"], self.config["cards"]
        self.dim, self.rm_fallback = self.config["dim"], self.config["rm_fallback"]
        self.game = GameConfig.from_dict(self.config["game"])
        self.num_actions = self.game.num_actions
        self.obs_dim = obs_dim_for(self.features)
        self.bet_index = np.asarray(bet_feature_indices(self.features, self.arch), dtype=np.int64)

        if self.cards == "embed":
            if self.arch == "paper":
                self.group_emb = [
                    (
                        w[f"card_model.group_embeddings.{g}.rank_embedding.weight"],
                        w[f"card_model.group_embeddings.{g}.suit_embedding.weight"],
                        w[f"card_model.group_embeddings.{g}.card_embedding.weight"],
                    )
                    for g in range(4)
                ]
            else:
                self.emb = (
                    w["card_model.cards_embeddings.rank_embedding.weight"],
                    w["card_model.cards_embeddings.suit_embedding.weight"],
                    w["card_model.cards_embeddings.card_embedding.weight"],
                )
            self.card_fc1 = (w["card_model.fc1.weight"], w["card_model.fc1.bias"])
        else:
            self.onehot = (w["card_model.onehot.weight"], w["card_model.onehot.bias"])
        self.card_fc = [(w[f"card_model.fc{i}.weight"], w[f"card_model.fc{i}.bias"]) for i in (2, 3)]
        if self.arch == "current":
            self.stage_emb = w["stage_and_order_model.stage_embedding.weight"]
            self.first_emb = w["stage_and_order_model.first_to_act_embedding.weight"]
            self.stage_fc = [
                (w[f"stage_and_order_model.fc{i}.weight"], w[f"stage_and_order_model.fc{i}.bias"]) for i in (1, 2)
            ]
        self.bets_fc = [(w[f"bets_model.fc{i}.weight"], w[f"bets_model.fc{i}.bias"]) for i in (1, 2)]
        self.comb = [(w[f"comb{i}.weight"], w[f"comb{i}.bias"]) for i in (1, 2, 3)]
        self.head = (w["action_head.weight"], w["action_head.bias"])

    @staticmethod
    def _linear(wb, x):
        w, b = wb
        return x @ w.T + b

    @staticmethod
    def _embed(tables, cards):  # cards (..., 3) -> (..., dim)
        rank, suit, card = tables
        return rank[cards[..., 0]] + suit[cards[..., 1]] + card[cards[..., 2]]

    def _card_branch(self, cards):  # (B, 7, 3) int
        if self.cards == "embed":
            if self.arch == "paper":
                g = self.group_emb
                x = np.concatenate(
                    [
                        self._embed(g[0], cards[:, 0]) + self._embed(g[0], cards[:, 1]),
                        self._embed(g[1], cards[:, 2]) + self._embed(g[1], cards[:, 3]) + self._embed(g[1], cards[:, 4]),
                        self._embed(g[2], cards[:, 5]),
                        self._embed(g[3], cards[:, 6]),
                    ],
                    axis=1,
                )
            else:
                emb = self._embed(self.emb, cards)
                x = np.concatenate([emb[:, :2].sum(axis=1), emb[:, 2:5].sum(axis=1), emb[:, 5], emb[:, 6]], axis=1)
            x = _relu(self._linear(self.card_fc1, x))
        else:
            # Linear on the concatenated one-hot cards == sum of the selected weight columns
            w, b = self.onehot
            cols = cards[:, :, 2] + CARD_CLASSES * np.arange(7)[None, :]  # (B, 7)
            x = _relu(w.T[cols].sum(axis=1) + b)
        for wb in self.card_fc:
            x = _relu(self._linear(wb, x))
        return x

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None]
        if obs.shape[1] < self.obs_dim:
            raise ValueError(f"observation has {obs.shape[1]} features, this network needs {self.obs_dim}")
        cards = obs[:, :21].astype(np.int64).reshape(-1, 7, 3)
        parts = [self._card_branch(cards)]
        if self.arch == "current":
            s = np.concatenate(
                [self.stage_emb[obs[:, 21].astype(np.int64)], self.first_emb[obs[:, 22].astype(np.int64)]], axis=1
            )
            for wb in self.stage_fc:
                s = _relu(self._linear(wb, s))
            parts.append(s)
        b = _relu(self._linear(self.bets_fc[0], obs[:, self.bet_index]))
        b = _relu(self._linear(self.bets_fc[1], b) + b)
        parts.append(b)
        z = np.concatenate(parts, axis=1)
        z = _relu(self._linear(self.comb[0], z))
        z = _relu(self._linear(self.comb[1], z) + z)
        z = _relu(self._linear(self.comb[2], z) + z)
        z = (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, ddof=1, keepdims=True) + 1e-6)
        out = self._linear(self.head, z)
        return out[0] if single else out
