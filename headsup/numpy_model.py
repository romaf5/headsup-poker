"""Pure-numpy mirror of :class:`headsup.model.BaseModel`.

CFR traversals evaluate the network on one observation at a time from many CPU worker
processes.  Doing that in PyTorch costs ~0.4 ms per call in framework overhead; the same
arithmetic in numpy on 1-D vectors takes ~40 µs and the workers do not need to import
torch at all.  ``NumpyModel.__call__`` handles both a single observation (31,) and a
batch (B, 31).
"""

import numpy as np


def _relu(x):
    return np.maximum(x, 0.0, out=x)


class NumpyModel:
    def __init__(self, weights: dict):
        w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.rank_emb = w["card_model.cards_embeddings.rank_embedding.weight"]
        self.suit_emb = w["card_model.cards_embeddings.suit_embedding.weight"]
        self.card_emb = w["card_model.cards_embeddings.card_embedding.weight"]
        self.card_fc = [
            (w[f"card_model.fc{i}.weight"], w[f"card_model.fc{i}.bias"]) for i in (1, 2, 3)
        ]
        self.stage_emb = w["stage_and_order_model.stage_embedding.weight"]
        self.first_emb = w["stage_and_order_model.first_to_act_embedding.weight"]
        self.stage_fc = [
            (w[f"stage_and_order_model.fc{i}.weight"], w[f"stage_and_order_model.fc{i}.bias"])
            for i in (1, 2)
        ]
        self.bets_fc = [(w[f"bets_model.fc{i}.weight"], w[f"bets_model.fc{i}.bias"]) for i in (1, 2)]
        self.comb = [(w[f"comb{i}.weight"], w[f"comb{i}.bias"]) for i in (1, 2, 3)]
        self.head = (w["action_head.weight"], w["action_head.bias"])

    @staticmethod
    def _linear(wb, x):
        w, b = wb
        return x @ w.T + b

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None]
        cards = obs[:, :21].astype(np.int64).reshape(-1, 7, 3)
        emb = self.rank_emb[cards[:, :, 0]] + self.suit_emb[cards[:, :, 1]] + self.card_emb[cards[:, :, 2]]
        x = np.concatenate(
            [emb[:, :2].sum(axis=1), emb[:, 2:5].sum(axis=1), emb[:, 5], emb[:, 6]], axis=1
        )
        for wb in self.card_fc:
            x = _relu(self._linear(wb, x))

        s = np.concatenate(
            [self.stage_emb[obs[:, 21].astype(np.int64)], self.first_emb[obs[:, 22].astype(np.int64)]],
            axis=1,
        )
        for wb in self.stage_fc:
            s = _relu(self._linear(wb, s))

        b = _relu(self._linear(self.bets_fc[0], obs[:, 23:31]))
        b = _relu(self._linear(self.bets_fc[1], b) + b)

        z = np.concatenate([x, s, b], axis=1)
        z = _relu(self._linear(self.comb[0], z))
        z = _relu(self._linear(self.comb[1], z) + z)
        z = _relu(self._linear(self.comb[2], z) + z)
        z = (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, ddof=1, keepdims=True) + 1e-6)
        out = self._linear(self.head, z)
        return out[0] if single else out
