"""The AlphaHoldem policy as a batched player (spec ``alpha:<policy.pth>``).

Stateless: the tensors are rebuilt from each observation row, so the same row always gets the same answer - what
``headsup.lbr`` needs (it queries the strategy on hand-substituted observations) and what lets one object sit at
any number of tables.  Actions are sampled from the policy (``deterministic``: the most likely one).
"""

import numpy as np
import torch

from headsup.alphaholdem.encoding import Encoder
from headsup.alphaholdem.model import load_alpha_net


class AlphaHoldemPlayer:
    def __init__(self, net, device=None, deterministic=False, seed=None, chunk=32768):
        """``net``: an :class:`AlphaNet` or the path of a file written by :meth:`AlphaNet.save`."""
        if isinstance(net, (str, bytes)) or hasattr(net, "__fspath__"):
            net = load_alpha_net(net, device=device)
        self.net = net
        self.game = net.game
        self.device = next(net.parameters()).device
        self.encoder = Encoder(self.game, self.device)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.chunk = int(chunk)
        self.last_probs = None

    @torch.no_grad()
    def probs(self, obs, ids=None):
        """float32[N, num_actions]: the policy at each observation row (0 for illegal actions)."""
        obs = np.asarray(obs, dtype=np.float32)
        out = np.empty((len(obs), self.game.num_actions), dtype=np.float32)
        training = self.net.training
        self.net.eval()
        for i in range(0, len(obs), self.chunk):
            cards, acts, legal = self.encoder(obs[i : i + self.chunk])
            logits, _ = self.net(cards, acts, legal)
            out[i : i + self.chunk] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        self.net.train(training)
        return out

    def __call__(self, obs, ids=None):
        self.last_probs = probs = self.probs(obs)
        if self.deterministic:
            return probs.argmax(axis=1).astype(np.int64)
        # u in (0, total]: an action of probability 0 (an illegal one) is never selected
        cum = np.cumsum(probs, axis=1, dtype=np.float64)
        u = (1.0 - self.rng.random((len(probs), 1))) * cum[:, -1:]
        return (u > cum).sum(axis=1).astype(np.int64)
