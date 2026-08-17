"""Game protocol shared by every game in the project.

A *state* is a node of the game tree: a chance node (``current_player == CHANCE``), a decision
node of player 0/1, or terminal.  Algorithms use only this interface, so tabular CFR, MCCFR,
Deep CFR / SD-CFR / DREAM / ESCHER, best-response computation and search work for every game
that implements it (Kuhn, Leduc, limit hold'em variants, the no-limit abstraction ...).

Conventions
- actions are ints ``0 .. num_actions-1``; ``legal_actions()`` lists the ones available;
- ``info_key(player)`` is a hashable identifier of the player's information set (tabular methods);
- ``info_state(player)`` is the float32 feature vector the networks consume (``game.obs_dim``);
- ``returns()`` are the two players' payoffs at a terminal state (zero-sum: ``r0 == -r1``);
- chance nodes are either enumerable (``chance_outcomes()`` -> [(action, prob)]) or only sampleable
  (``sample_chance(rng)``), games say which via ``enumerable_chance``.
"""

from dataclasses import dataclass

import numpy as np

CHANCE = -1
TERMINAL = -2


class State:
    """Mutable game state; ``clone()`` must be cheap."""

    current_player: int

    def clone(self):
        raise NotImplementedError

    def legal_actions(self):
        raise NotImplementedError

    def legal_mask(self):
        mask = np.zeros(self.game.num_actions, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def apply(self, action):
        raise NotImplementedError

    def child(self, action):
        c = self.clone()
        c.apply(action)
        return c

    def is_terminal(self):
        return self.current_player == TERMINAL

    def is_chance(self):
        return self.current_player == CHANCE

    def returns(self):
        raise NotImplementedError

    def chance_outcomes(self):
        raise NotImplementedError

    def sample_chance(self, rng):
        outcomes = self.chance_outcomes()
        idx = rng.choice(len(outcomes), p=[p for _, p in outcomes])
        return outcomes[idx][0]

    def info_key(self, player):
        raise NotImplementedError

    def info_state(self, player):
        raise NotImplementedError

    def public_key(self):
        """Hashable id of the public state (betting sequence + public cards); default: none."""
        return None

    def history_key(self):
        """Hashable id of the full history (all players' information); default: both info keys."""
        return (self.info_key(0), self.info_key(1))


class Game:
    name = "game"
    num_players = 2
    num_actions = 0
    obs_dim = 0
    enumerable_chance = True

    def new_initial_state(self) -> State:
        raise NotImplementedError

    def make_model(self, **kwargs):
        """A torch module mapping (B, obs_dim) -> (B, num_actions) logits / advantages."""
        raise NotImplementedError

    def action_names(self):
        return [str(a) for a in range(self.num_actions)]

    # -- helpers ------------------------------------------------------------------------
    def playout(self, policies, rng, state=None):
        """Play one hand with ``policies[p](state) -> probs``; returns the terminal state."""
        s = state.clone() if state is not None else self.new_initial_state()
        while not s.is_terminal():
            if s.is_chance():
                s.apply(s.sample_chance(rng))
            else:
                probs = np.asarray(policies[s.current_player](s), dtype=np.float64)
                s.apply(int(rng.choice(len(probs), p=probs / probs.sum())))
        return s


@dataclass
class UniformPolicy:
    game: Game

    def __call__(self, state):
        p = np.zeros(self.game.num_actions)
        p[state.legal_actions()] = 1.0
        return p / p.sum()
