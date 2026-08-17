"""Two-player heads-up Texas Hold'em engine with a configurable coarse action set.

Rules
-----
* Seat 0 is the dealer / small blind and acts first pre-flop; seat 1 posts the big blind
  and acts first on every later street.  Stacks are reset every hand.
* Actions: FOLD, CHECK_CALL, one RAISE per configured bet size (``GameConfig.bet_sizes``:
  ``"min"`` = call + one big blind, or a fraction of the pot after calling), ALL_IN.
  A raise that the player cannot afford becomes an all-in.  The ``raise_cap``-th raise
  in an uninterrupted sequence of raises is converted into an all-in, which keeps the
  game tree finite for CFR (identical to the rule the DeepCFR models were trained with).
  ``legal_mask()`` tells which actions are meaningful (fold only facing a bet; with
  ``mask_redundant`` also no raise that merely duplicates another action).
* Once bets are matched and any player is all-in, the remaining board is dealt and the
  hand goes to showdown.  Rewards are chips won/lost by each seat (zero-sum).

The engine is deliberately dependency-light and cheap to ``clone()`` so that CFR
traversals can branch on it.
"""

import numpy as np

from headsup.cards import CARD_FEATURES, NUM_CARDS, hand_strength
from headsup.enums import Action, Stage
from headsup.game import DEFAULT_GAME, GameConfig

# Observation layout (float32[OBS_DIM]); the first OBS_DIM_AGGREGATED entries are the original
# 31-feature encoding, the rest is the per-street bet history of DeepCFR (Brown et al. 2019):
# for each of the HISTORY_ROUNDS betting rounds and each of its first HISTORY_SLOTS actions,
# [chips put in by that action / pot before it, 1.0 (an action occurred)] (0, 0 = no action).
OBS_DIM_AGGREGATED = 31
HISTORY_ROUNDS = 4
HISTORY_SLOTS = 6  # DeepCFR: "in each betting round there can be at most 6 sequential actions"
HISTORY_OFFSET = OBS_DIM_AGGREGATED
HISTORY_DIM = HISTORY_ROUNDS * HISTORY_SLOTS * 2
OBS_DIM_HISTORY = HISTORY_OFFSET + HISTORY_DIM  # 79: what history-feature networks read
RAISES_INDEX = OBS_DIM_HISTORY  # [79] consecutive raises on this street (the raise-cap counter)
OBS_DIM = OBS_DIM_HISTORY + 1  # 80
BOARD_CARDS_BY_STAGE = (0, 3, 4, 5, 5)  # PREFLOP, FLOP, TURN, RIVER, END


def history_slot(round_index, k):
    """Index of the ``[size, occurred]`` pair of the k-th action of a betting round."""
    return HISTORY_OFFSET + 2 * (HISTORY_SLOTS * round_index + k)


def public_state_from_obs(obs):
    """(to_call, pot, stack, consecutive_raises) as ints from observation rows (N, >= 80)."""
    obs = np.asarray(obs)
    pot = np.rint(obs[:, 29] * 1000).astype(np.int64)
    to_call = np.rint(obs[:, 23] * pot).astype(np.int64)
    stack = np.rint(obs[:, 28] * pot).astype(np.int64)
    raises = np.rint(obs[:, RAISES_INDEX]).astype(np.int64)
    return to_call, pot, stack, raises


def legal_mask_from_obs(obs, game=DEFAULT_GAME):
    """bool[N, num_actions]: the engine's ``legal_mask`` recomputed from observations."""
    to_call, pot, stack, raises = public_state_from_obs(obs)
    return np.array([game.legal_mask(int(c), int(p), int(s), int(r)) for c, p, s, r in zip(to_call, pot, stack, raises)], dtype=bool)


class HeadsUpPoker:
    NUM_PLAYERS = 2

    def __init__(
        self,
        stack_size: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
        raise_cap: int = 3,
        bet_sizes=("min",),
        mask_redundant: bool = False,
        rng: np.random.Generator | None = None,
        game: GameConfig | None = None,
    ):
        self.game = game if game is not None else GameConfig(stack_size, small_blind, big_blind, raise_cap, bet_sizes, mask_redundant)
        self.stack_size = self.game.stack_size
        self.small_blind = self.game.small_blind
        self.big_blind = self.game.big_blind
        self.raise_cap = self.game.raise_cap
        self.num_actions = self.game.num_actions
        self.all_in = self.game.all_in_action  # index of the all-in action, or None (limit games)
        self.num_rounds = self.game.num_rounds
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
        # bet history: per street, size (chips / pot before the action) of the first HISTORY_SLOTS
        # actions and the number of actions taken on that street
        self.history_size = [[0.0] * HISTORY_SLOTS for _ in range(HISTORY_ROUNDS)]
        self.history_n = [0] * HISTORY_ROUNDS

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
        self.history_size = [[0.0] * HISTORY_SLOTS for _ in range(HISTORY_ROUNDS)]
        self.history_n = [0] * HISTORY_ROUNDS
        return self.observation()

    def clone(self):
        """Cheap copy for tree search (cards are shared, chip state is copied)."""
        other = HeadsUpPoker.__new__(HeadsUpPoker)
        other.__dict__.update(self.__dict__)
        other.stacks = self.stacks.copy()
        other.bets = self.bets.copy()
        other.stage_bets = self.stage_bets.copy()
        other.rewards = self.rewards.copy()
        other.history_size = [row.copy() for row in self.history_size]
        other.history_n = self.history_n.copy()
        return other

    # ------------------------------------------------------------------ queries
    @property
    def visible_board(self):
        return self.board[: BOARD_CARDS_BY_STAGE[self.stage]]

    @property
    def to_call(self):
        p = self.current
        return self.stage_bets[1 - p] - self.stage_bets[p]

    @property
    def fold_allowed(self):
        """Folding only exists when facing a bet (with nothing to call it would be a dominated check)."""
        return self.to_call > 0

    def raise_amount(self, action):
        """Chips the current player puts in for raise ``action`` (2 .. 2+K-1), capped by the stack."""
        return self.game.raise_amount(action, self.to_call, self.pot, self.stacks[self.current], int(self.stage))

    def legal_mask(self):
        """bool per action (see :meth:`GameConfig.legal_mask`)."""
        return self.game.legal_mask(self.to_call, self.pot, self.stacks[self.current], self.consecutive_raises, round_index=int(self.stage))

    def legal_mask_and_twins(self):
        """(mask, twins): ``twins[a]`` is the action a redundant ``a`` duplicates (else ``a``)."""
        return self.game.legal_mask(self.to_call, self.pot, self.stacks[self.current], self.consecutive_raises, with_twins=True, round_index=int(self.stage))

    def legal_actions(self):
        return [a for a, ok in enumerate(self.legal_mask()) if ok]

    def observation(self, seat=None):
        """Observation vector (float32[OBS_DIM]) from the point of view of ``seat``.

        Layout: hand (2 x [rank+1, suit+1, card+1], sorted), board (5 x same, flop sorted, 0-padded), stage,
        first_to_act_next_stage, 8 normalised bet/stack features (the original 31 features),
        then the bet history: HISTORY_ROUNDS x HISTORY_SLOTS x [size / pot, occurred].
        Networks trained on the 31-feature layout simply read the first 31 entries.
        """
        p = self.current if seat is None else seat
        o = 1 - p
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # canonical card order (hole cards and flop sorted by id): permutation-invariant networks
        # do not care, one-hot ones and LBR's hand substitution rely on it
        obs[0:6] = CARD_FEATURES[sorted(self.hands[p])].ravel()
        n_board = BOARD_CARDS_BY_STAGE[self.stage]
        if n_board:
            cards = sorted(self.board[:3]) + list(self.board[3:n_board])
            obs[6 : 6 + 3 * n_board] = CARD_FEATURES[cards].ravel()
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
        obs[RAISES_INDEX] = self.consecutive_raises
        for r in range(HISTORY_ROUNDS):
            n = self.history_n[r]
            if n:
                base = history_slot(r, 0)
                sizes = self.history_size[r]
                for k in range(min(n, HISTORY_SLOTS)):
                    obs[base + 2 * k] = sizes[k]
                    obs[base + 2 * k + 1] = 1.0
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
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action {action}")
        p = self.current
        o = 1 - p

        if action == Action.FOLD and self.stage_bets[o] == self.stage_bets[p]:
            action = Action.CHECK_CALL  # nothing to call: folding is a (dominated) check

        raise_amount = None
        if self.game.is_raise(action):
            raise_amount = self.raise_amount(action)
            self.consecutive_raises += 1
            cap = self.game.cap(int(self.stage))
            if self.all_in is not None and self.consecutive_raises >= cap:
                action = self.all_in  # no-limit: the cap-th raise in a row becomes an all-in
            elif self.all_in is None and self.consecutive_raises > cap:
                self.consecutive_raises -= 1  # limit: no raise past the cap - executed as a call
                action = Action.CHECK_CALL
        else:
            self.consecutive_raises = 0

        if action == Action.FOLD:
            self._record(0)
            self.folded = p
            self.rewards[o] = self.bets[p]
            self.rewards[p] = -self.bets[p]
            self.done = True
            return self.observation(p), self.rewards, True, {}

        if action == Action.CHECK_CALL:
            amount = min(self.stage_bets[o] - self.stage_bets[p], self.stacks[p])
        elif self.all_in is not None and action == self.all_in:
            amount = self.stacks[p]
        else:  # RAISE_k
            amount = raise_amount

        self._record(amount)
        self.bets[p] += amount
        self.stage_bets[p] += amount
        self.stacks[p] -= amount
        self.pot += amount
        self.acted |= 1 << p
        self.current = o

        if self._street_finished():
            if int(self.stage) == self.num_rounds - 1 or min(self.stacks) == 0:
                self._showdown()
                return self.observation(p), self.rewards, True, {}
            self._next_street()
        return self.observation(), None, False, {}

    def _record(self, amount):
        """Append the current action (``amount`` chips into the current pot) to the street's history."""
        r = int(self.stage)
        k = self.history_n[r]
        if k < HISTORY_SLOTS:
            self.history_size[r][k] = amount / self.pot
        self.history_n[r] = k + 1

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
        # showdown on the board of the game's last betting round (all-ins run the board out; FHP: 3 cards)
        board = self.board[: BOARD_CARDS_BY_STAGE[self.num_rounds - 1]]
        self.stage = Stage.END
        s0 = hand_strength(self.hands[0], board)
        s1 = hand_strength(self.hands[1], board)
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
            action = int(input(f"action (0 fold, 1 check/call, 2.. raise sizes {engine.game.bet_sizes}, all-in {engine.all_in}, q quit): "))
        except ValueError:
            break
        _, rewards, done, _ = engine.step(action)
        if done:
            print(engine.describe())
            print("rewards:", rewards)
            engine.reset()


if __name__ == "__main__":
    play_interactive()
