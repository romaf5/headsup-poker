"""AlphaHoldem's pseudo-siamese network: one ConvNet per input tensor (no shared parameters), fused by fully
connected layers into a policy head and a value head (the paper's Fig. 2).

The paper gives no layer list (8.6 M parameters: 1.8 M in the ConvNets, 6.8 M in the FC layers, for a 9-action game);
here each tower is ``conv_layers`` x [3x3 convolution with ``channels`` filters, ReLU] on the 4 x 13 card planes /
the 4 x A action planes, flattened into a ``hidden``-unit layer; the two are concatenated into one more ``hidden``
layer that feeds both heads (defaults 3 / 64 / 256: 1.4 M parameters in the 4-action game).  No normalisation
layers.  Logits of illegal actions are set to ``MASKED`` so that the softmax gives them probability 0.

Saved as ``{"config", "state_dict"}`` like :class:`headsup.model.BaseModel`; ``config["game"]`` is the action tree.
"""

import torch
import torch.nn as nn

from headsup.alphaholdem.encoding import ACTION_CHANNELS, ACTION_ROWS, CARD_CHANNELS
from headsup.game import DEFAULT_GAME, GameConfig

MASKED = -1e9  # logit of an illegal action (finite: 0 * log 0 stays 0 in the entropy)


def _tower(in_channels, channels, conv_layers, cells, hidden):
    layers = []
    for i in range(conv_layers):
        layers += [nn.Conv2d(in_channels if i == 0 else channels, channels, kernel_size=3, padding=1), nn.ReLU()]
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(channels * cells, hidden), nn.ReLU())


class AlphaNet(nn.Module):
    def __init__(self, game=DEFAULT_GAME, channels=64, conv_layers=3, hidden=256, config=None):
        super().__init__()
        if config is not None:
            game = GameConfig.from_dict(config["game"])
            channels, conv_layers, hidden = int(config["channels"]), int(config["conv_layers"]), int(config["hidden"])
        self.game = game
        self.config = dict(kind="alphaholdem", game=game.tree_dict(), channels=channels, conv_layers=conv_layers, hidden=hidden)
        self.num_actions = game.num_actions
        self.card_tower = _tower(CARD_CHANNELS, channels, conv_layers, 4 * 13, hidden)
        self.action_tower = _tower(ACTION_CHANNELS, channels, conv_layers, ACTION_ROWS * self.num_actions, hidden)
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, self.num_actions)
        self.value_head = nn.Linear(hidden, 1)
        for m in self.modules():  # the usual PPO initialisation (the paper does not say)
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, cards, actions, legal=None):
        """(logits[B, A], value[B]) from the card tensor [B, 6, 4, 13] and the action tensor [B, 24, 4, A] (bool or
        float); with ``legal`` (bool[B, A]) the logits of the other actions are ``MASKED``."""
        dtype = self.policy_head.weight.dtype
        z = self.fuse(torch.cat([self.card_tower(cards.to(dtype)), self.action_tower(actions.to(dtype))], dim=1))
        logits = self.policy_head(z)
        if legal is not None:
            logits = logits.masked_fill(~legal, MASKED)
        return logits, self.value_head(z).squeeze(-1)

    def parameter_counts(self):
        conv = sum(p.numel() for m in self.modules() if isinstance(m, nn.Conv2d) for p in m.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"conv": conv, "fc": total - conv, "total": total}

    def save(self, path):
        torch.save({"config": dict(self.config), "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}, path)

    @staticmethod
    def from_state(state):
        """Build a network from ``{"config", "state_dict"}`` (the :meth:`save` format)."""
        if state.get("config", {}).get("kind") != "alphaholdem":
            raise ValueError("not an AlphaHoldem model file (config kind != 'alphaholdem')")
        net = AlphaNet(config=state["config"])
        net.load_state_dict(state["state_dict"])
        return net


def load_alpha_net(path, device=None) -> AlphaNet:
    """Load a network written by :meth:`AlphaNet.save` (in eval mode)."""
    net = AlphaNet.from_state(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net.to(device) if device is not None else net
