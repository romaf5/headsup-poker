"""K-Best self-play pool (AlphaHoldem, "Efficient Model Selection and Generation").

The paper: "a pool of competing agents from the historical versions of the main agent ... by competing among
different agents, the algorithm selects the K best survivors from their ELO scores and generates experience replays
simultaneously".  It gives neither K, the snapshot interval, the opponent sampling nor how ELO is computed from
poker results.  Here:

* a snapshot (a frozen copy of the main network) enters with the main agent's current rating; the pool then keeps
  the ``k`` members with the highest rating (of equal ones the older leaves);
* a game is the block of hands the main agent played against one member in one iteration, won / drawn / lost by
  the sign of its chip total (a per-hand win / loss would ignore the pot sizes); both ratings move by the standard
  ELO update (scale 400, ``k_factor``);
* pool members only play the main agent; every iteration each table gets the current agent (-1) or a member as
  opponent, uniformly (the same number of tables each, in random order).
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch


def elo_expected(rating, other):
    """Expected score of a player rated ``rating`` against one rated ``other``."""
    return 1.0 / (1.0 + 10.0 ** ((other - rating) / 400.0))


@dataclass
class Member:
    net: torch.nn.Module  # frozen copy of the main network
    elo: float
    iteration: int


class KBestPool:
    def __init__(self, k=8, k_factor=16.0, initial_elo=1200.0):
        self.k, self.k_factor = int(k), float(k_factor)
        self.main_elo = float(initial_elo)
        self.members = []

    def __len__(self):
        return len(self.members)

    def add(self, net, iteration):
        """Snapshot ``net`` as a new member; returns the iteration of the member that had to leave (None: nobody)."""
        frozen = copy.deepcopy(net).eval().requires_grad_(False)
        self.members.append(Member(frozen, self.main_elo, int(iteration)))
        if len(self.members) <= self.k:
            return None
        weakest = min(range(len(self.members)), key=lambda i: (self.members[i].elo, self.members[i].iteration))
        return self.members.pop(weakest).iteration

    def record(self, index, chips, hands):
        """One game of the main agent against member ``index``: its chip total over ``hands`` hands."""
        if hands <= 0:
            return
        member = self.members[index]
        score = 1.0 if chips > 0 else 0.0 if chips < 0 else 0.5
        delta = self.k_factor * (score - elo_expected(self.main_elo, member.elo))
        self.main_elo += delta
        member.elo -= delta

    def assign(self, num_tables, rng):
        """int64[num_tables]: each table's opponent for an iteration, -1 = the current agent, else a member index."""
        opponents = np.arange(num_tables) % (len(self.members) + 1) - 1
        return rng.permutation(opponents)

    def state_dict(self):
        return {
            "main_elo": self.main_elo,
            "config": dict(self.members[0].net.config) if self.members else None,
            "members": [{"elo": m.elo, "iteration": m.iteration, "state_dict": {k: v.detach().cpu() for k, v in m.net.state_dict().items()}}
                        for m in self.members],
        }

    def load_state_dict(self, state, device=None):
        from headsup.alphaholdem.model import AlphaNet

        self.main_elo = float(state["main_elo"])
        self.members = []
        for m in state["members"]:
            net = AlphaNet(config=state["config"])
            net.load_state_dict(m["state_dict"])
            net = net.to(device) if device is not None else net
            self.members.append(Member(net.eval().requires_grad_(False), float(m["elo"]), int(m["iteration"])))
