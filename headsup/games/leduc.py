"""Kuhn poker and Leduc hold'em (Southey et al. 2005), the standard small test games.

Leduc: 6-card deck (2 suits x J Q K), each player antes 1 and gets one private card; a betting
round (bets of 2), then one public card and a second round (bets of 4); at most 2 raises per
round; showdown: a pair (private = public rank) wins, else the higher rank, ties split.
Actions: 0 fold, 1 check/call, 2 bet/raise.  Kuhn: 3-card deck (J Q K), ante 1, one round with
a single bet of 1, no raise on a bet.

Information state features (Leduc): private card one-hot (3 ranks; suits are strategically
irrelevant but kept as separate cards in the deck), public card one-hot (3, zeros before the
flop), round, chips put in by each player this round / 4, pot / 20, and the round's action
history (up to 4 slots x 3 actions one-hot) - 3 + 3 + 1 + 2 + 1 + 12 = 22 features.
"""

import numpy as np

from headsup.games.base import CHANCE, TERMINAL, Game, State

FOLD, CALL, RAISE = 0, 1, 2


class LeducState(State):
    __slots__ = ("game", "current_player", "cards", "board", "round", "pot", "bets", "raises", "history", "folded", "deck")

    def __init__(self, game):
        self.game = game
        self.current_player = CHANCE
        self.cards = [None, None]
        self.board = None
        self.round = 0
        self.pot = 2 * game.ante
        self.bets = [0, 0]  # chips put in this round
        self.raises = 0
        self.history = [[], []]  # actions per round
        self.folded = None
        self.deck = list(range(game.deck_size))

    def clone(self):
        c = LeducState.__new__(LeducState)
        c.game = self.game
        c.current_player = self.current_player
        c.cards = list(self.cards)
        c.board = self.board
        c.round = self.round
        c.pot = self.pot
        c.bets = list(self.bets)
        c.raises = self.raises
        c.history = [list(self.history[0]), list(self.history[1])]
        c.folded = self.folded
        c.deck = list(self.deck)
        return c

    # -- chance -------------------------------------------------------------------------
    def chance_outcomes(self):
        p = 1.0 / len(self.deck)
        return [(c, p) for c in self.deck]

    def _deal(self, card):
        self.deck.remove(card)
        if self.cards[0] is None:
            self.cards[0] = card
        elif self.cards[1] is None:
            self.cards[1] = card
            self.current_player = 0
        else:
            self.board = card
            self.current_player = 0

    # -- betting ------------------------------------------------------------------------
    def to_call(self):
        p = self.current_player
        return self.bets[1 - p] - self.bets[p]

    def legal_actions(self):
        if self.current_player < 0:
            return []
        acts = [CALL]
        if self.to_call() > 0:
            acts.insert(0, FOLD)
        if self.raises < self.game.max_raises[self.round]:
            acts.append(RAISE)
        return acts

    def apply(self, action):
        if self.current_player == CHANCE:
            self._deal(action)
            return
        p = self.current_player
        self.history[self.round].append(action)
        if action == FOLD:
            self.folded = p
            self.current_player = TERMINAL
            return
        bet = self.game.bet_sizes[self.round]
        if action == CALL:
            amount = self.to_call()
            self.bets[p] += amount
            self.pot += amount
            # a call closes the round unless it is the first action of the round (a check)
            round_over = len(self.history[self.round]) > 1 or amount > 0
            if not round_over:
                self.current_player = 1 - p
                return
            self._next_round()
            return
        # raise / bet
        amount = self.to_call() + bet
        self.bets[p] += amount
        self.pot += amount
        self.raises += 1
        self.current_player = 1 - p

    def _next_round(self):
        if self.round + 1 >= self.game.num_rounds:
            self.current_player = TERMINAL
            return
        self.round += 1
        self.bets = [0, 0]
        self.raises = 0
        self.current_player = CHANCE if self.game.board_rounds and self.round in self.game.board_rounds else 0

    # -- outcome ------------------------------------------------------------------------
    def returns(self):
        assert self.current_player == TERMINAL
        invested = self.game.ante + sum(self.history_chips(p) for p in ()) if False else None  # placeholder for clarity
        # chips each player put in: ante + bets over rounds -> recompute from the pot symmetric split
        contrib = self._contributions()
        if self.folded is not None:
            w = 1 - self.folded
            r = [0.0, 0.0]
            r[w] = contrib[self.folded]
            r[self.folded] = -contrib[self.folded]
            return r
        s = [self.game.hand_value(self.cards[p], self.board) for p in range(2)]
        if s[0] == s[1]:
            return [0.0, 0.0]
        w = 0 if s[0] > s[1] else 1
        r = [0.0, 0.0]
        r[w] = contrib[1 - w]
        r[1 - w] = -contrib[1 - w]
        return r

    def _contributions(self):
        c = [self.game.ante, self.game.ante]
        # replay bets from the histories
        for rnd in range(self.game.num_rounds):
            b = self.game.bet_sizes[rnd]
            put = [0, 0]
            p = 0
            for a in self.history[rnd]:
                if a == CALL:
                    put[p] += put[1 - p] - put[p]
                elif a == RAISE:
                    put[p] += put[1 - p] - put[p] + b
                p = 1 - p
            c[0] += put[0]
            c[1] += put[1]
        return c

    # -- information ----------------------------------------------------------------------
    def info_key(self, player):
        return (player, self.cards[player], self.board, self.round, tuple(tuple(h) for h in self.history))

    def public_key(self):
        return (self.board, self.round, tuple(tuple(h) for h in self.history))

    def info_state(self, player):
        g = self.game
        x = np.zeros(g.obs_dim, dtype=np.float32)
        x[g.rank_of(self.cards[player])] = 1.0
        if self.board is not None:
            x[3 + g.rank_of(self.board)] = 1.0
        x[6] = self.round
        x[7] = self.bets[player] / 4.0
        x[8] = self.bets[1 - player] / 4.0
        x[9] = self.pot / 20.0
        off = 10
        for k, a in enumerate(self.history[self.round][:4]):
            x[off + 3 * k + a] = 1.0
        return x


class Leduc(Game):
    name = "leduc"
    num_actions = 3
    obs_dim = 22
    ante = 1
    deck_size = 6
    num_rounds = 2
    board_rounds = (1,)
    bet_sizes = (2, 4)
    max_raises = (2, 2)

    def rank_of(self, card):
        return card % 3  # cards 0..5 = J,Q,K x 2 suits

    def hand_value(self, card, board):
        r = self.rank_of(card)
        return 10 + r if board is not None and self.rank_of(board) == r else r

    def new_initial_state(self):
        return LeducState(self)

    def action_names(self):
        return ["fold", "call", "raise"]

    def make_model(self, hidden=64, layers=3, in_dim=None):
        import torch.nn as nn

        mods, d = [], (in_dim or self.obs_dim)
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        head = nn.Linear(d, self.num_actions)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return nn.Sequential(*mods, head)


class Kuhn(Leduc):
    """Kuhn poker: 3 cards, ante 1, one round, one bet of 1, no re-raise."""

    name = "kuhn"
    obs_dim = 22
    deck_size = 3
    num_rounds = 1
    board_rounds = ()
    bet_sizes = (1,)
    max_raises = (1,)

    def rank_of(self, card):
        return card

    def hand_value(self, card, board):
        return card
