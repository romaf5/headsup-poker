"""Action / stage enums.  ``Action`` names the indices of the default 4-action game
(fold, check/call, min-raise, all-in); games with several bet sizes have more raise indices and
their all-in is ``GameConfig.all_in`` (see headsup/game.py)."""

from enum import IntEnum


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    ALL_IN = 3


class Stage(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END = 4


NUM_ACTIONS = len(Action)  # of the default game
