"""Two-player heads-up Texas Hold'em engine with a coarse action set.

Rules
-----
* Seat 0 is the dealer / small blind and acts first pre-flop; seat 1 posts the big blind
  and acts first on every later street.  Stacks are reset every hand.
* Actions: FOLD, CHECK_CALL, RAISE (min-raise: call + one big blind), ALL_IN.
  A raise that the player cannot afford becomes an all-in.  The ``raise_cap``-th raise
  in an uninterrupted sequence of raises is converted into an all-in, which keeps the
  game tree finite for CFR (identical to the rule the DeepCFR models were trained with).
* Once bets are matched and any player is all-in, the remaining board is dealt and the
  hand goes to showdown.  Rewards are chips won/lost by each seat (zero-sum).

The engine is deliberately dependency-light and cheap to ``clone()`` so that CFR
traversals can branch on it.
"""

import numpy as np

from headsup.cards import CARD_FEATURES, NUM_CARDS, hand_strength
from headsup.enums import NUM_ACTIONS, Action, Stage

OBS_DIM = 31
BOARD_CARDS_BY_STAGE = (0, 3, 4, 5, 5)  # PREFLOP, FLOP, TURN, RIVER, END


class HeadsUpPoker:
    NUM_PLAYERS = 2

    def __init__(
        self,
        stack_size: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
        raise_cap: int = 3,
        rng: np.random.Generator | None = None,
    ):
        assert 0 < small_blind < big_blind < stack_size
        self.stack_size = stack_size
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.raise_cap = raise_cap
        self.rng = rng if rng is not None else np.random.default_rng()
        self.dealer = 0
        self.hands_played = 0

        # per-hand state (initialised in reset)
        self.hands = ((0, 1), (2, 3))
        self.board = (4, 5, 6, 7, 8)
        self.stacks = [0, 0]
        self.bets = [0, 0]  # total chips committed this hand
        self.stage_bets = [0, 0]  # chips committed on the current street
        self.pot = 0
        self.stage = Stage.PREFLOP
        self.current = 0
        self.folded = -1  # seat that folded, or -1
        self.acted = 0  # bitmask of seats that acted on this street
        self.consecutive_raises = 0
        self.done = True
        self.rewards = [0, 0]

    # ------------------------------------------------------------------ dealing
    def reset(self, deck=None):
        """Start a new hand and return the observation of the player to act (seat 0)."""
        self.hands_played += 1
        if deck is None:
            deck = self.rng.permutation(NUM_CARDS)
        deck = [int(c) for c in deck[:9]]
        self.hands = ((deck[0], deck[1]), (deck[2], deck[3]))
        self.board = tuple(deck[4:9])

        self.stacks = [self.stack_size - self.small_blind, self.stack_size - self.big_blind]
        self.bets = [self.small_blind, self.big_blind]
        self.stage_bets = [self.small_blind, self.big_blind]
        self.pot = self.small_blind + self.big_blind
        self.stage = Stage.PREFLOP
        self.current = self.dealer
        self.folded = -1
        self.acted = 0
        self.consecutive_raises = 0
        self.done = False
        self.rewards = [0, 0]
        return self.observation()

    def clone(self):
        """Cheap copy for tree search (cards are shared, chip state is copied)."""
        other = HeadsUpPoker.__new__(HeadsUpPoker)
        other.__dict__.update(self.__dict__)
        other.stacks = self.stacks.copy()
        other.bets = self.bets.copy()
        other.stage_bets = self.stage_bets.copy()
        other.rewards = self.rewards.copy()
        return other

    # ------------------------------------------------------------------ queries
    @property
    def visible_board(self):
        return self.board[: BOARD_CARDS_BY_STAGE[self.stage]]

    def legal_actions(self):
        """All four actions are always accepted; kept for API completeness."""
        return list(Action)

    def observation(self, seat=None):
        """Observation vector (float32[31]) from the point of view of ``seat``.

        Layout: hand (2 x [rank+1, suit+1, card+1]), board (5 x same, 0-padded),
        stage, first_to_act_next_stage, 8 normalised bet/stack features.
        """
        p = self.current if seat is None else seat
        o = 1 - p
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:6] = CARD_FEATURES[list(self.hands[p])].ravel()
        n_board = BOARD_CARDS_BY_STAGE[self.stage]
        if n_board:
            obs[6 : 6 + 3 * n_board] = CARD_FEATURES[list(self.board[:n_board])].ravel()
        obs[21] = int(self.stage)
        obs[22] = p != self.dealer

        pot = self.pot
        stack = self.stacks[p]
        diff = self.stage_bets[o] - self.stage_bets[p]
        obs[23] = diff / pot
        obs[24] = self.bets[p] / pot
        obs[25] = self.bets[o] / pot
        obs[26] = self.stage_bets[p] / pot
        obs[27] = self.stage_bets[o] / pot
        obs[28] = stack / pot
        obs[29] = pot / 1000
        obs[30] = diff / stack if stack > 0 else 0.0
        return obs

    # ------------------------------------------------------------------ actions
    def step(self, action):
        """Apply ``action`` for the current player.

        Returns ``(obs, rewards, done, info)`` where ``obs`` is the next player's
        observation (or the acting player's terminal observation when done) and
        ``rewards`` is the per-seat chip result (only meaningful when done).
        """
        assert not self.done, "call reset() first"
        action = int(action)
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"Invalid action {action}")
        p = self.current
        o = 1 - p

        if action == Action.RAISE:
            self.consecutive_raises += 1
            if self.consecutive_raises >= self.raise_cap:
                action = Action.ALL_IN
        else:
            self.consecutive_raises = 0

        if action == Action.FOLD:
            self.folded = p
            self.rewards[o] = self.bets[p]
            self.rewards[p] = -self.bets[p]
            self.done = True
            return self.observation(p), self.rewards, True, {}

        if action == Action.CHECK_CALL:
            amount = min(self.stage_bets[o] - self.stage_bets[p], self.stacks[p])
        elif action == Action.RAISE:
            amount = min(
                self.stage_bets[o] - self.stage_bets[p] + self.big_blind, self.stacks[p]
            )
        else:  # ALL_IN
            amount = self.stacks[p]

        self.bets[p] += amount
        self.stage_bets[p] += amount
        self.stacks[p] -= amount
        self.pot += amount
        self.acted |= 1 << p
        self.current = o

        if self._street_finished():
            if self.stage == Stage.RIVER or min(self.stacks) == 0:
                self._showdown()
                return self.observation(p), self.rewards, True, {}
            self._next_street()
        return self.observation(), None, False, {}

    def _street_finished(self):
        if self.acted != 0b11:
            return False
        if self.stage_bets[0] == self.stage_bets[1]:
            return True
        # unmatched bet is fine only if the short player is all-in
        short = 0 if self.stage_bets[0] < self.stage_bets[1] else 1
        return self.stacks[short] == 0

    def _next_street(self):
        self.stage = Stage(self.stage + 1)
        self.stage_bets = [0, 0]
        self.acted = 0
        self.consecutive_raises = 0
        self.current = 1 - self.dealer  # big blind acts first post-flop

    def _showdown(self):
        self.stage = Stage.END
        s0 = hand_strength(self.hands[0], self.board)
        s1 = hand_strength(self.hands[1], self.board)
        won = min(self.bets)  # excess of an uncalled all-in is returned
        if s0 == s1:
            self.rewards = [0, 0]
        elif s0 < s1:
            self.rewards = [won, -won]
        else:
            self.rewards = [-won, won]
        self.done = True

    # ------------------------------------------------------------------ debugging
    def describe(self, seat=None):
        from headsup.cards import card_to_str

        p = self.current if seat is None else seat
        lines = [
            f"hand #{self.hands_played} stage={self.stage.name} to_act=seat{self.current}",
            f"  seat{p} hand: {[card_to_str(c) for c in self.hands[p]]}",
            f"  board: {[card_to_str(c) for c in self.visible_board]}",
            f"  stacks={self.stacks} bets={self.bets} street_bets={self.stage_bets} pot={self.pot}",
        ]
        if self.done:
            lines.append(f"  DONE rewards={self.rewards} folded={self.folded}")
        return "\n".join(lines)


def play_interactive():
    """Tiny CLI to sanity check the rules by hand."""
    engine = HeadsUpPoker()
    engine.reset()
    while True:
        print(engine.describe())
        try:
            action = int(input("action (0 fold, 1 check/call, 2 raise, 3 all-in, q quit): "))
        except ValueError:
            break
        _, rewards, done, _ = engine.step(action)
        if done:
            print(engine.describe())
            print("rewards:", rewards)
            engine.reset()


if __name__ == "__main__":
    play_interactive()
