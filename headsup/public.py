"""Public-state reconstruction: rebuild an engine (with the hero's cards, dummy cards for the
opponent) from a single observation by replaying the bet history it contains.

Players only ever see observations, but range tracking (LBR-style Bayesian updates) and
subgame search need the sequence of public actions and the opponent's observations at each of
its decisions.  The observation carries everything needed: the board, the position, and per
street the first HISTORY_SLOTS actions as [chips put in / pot before, occurred].  Replaying
those amounts from the blinds on a fresh engine reproduces the public state exactly (chips are
integers, the pot before every action is known during the replay, and the action that produced a
given amount is unique up to the raise / all-in conversion, which does not change later play).
"""

import numpy as np

from headsup.cards import NUM_CARDS
from headsup.engine import HISTORY_ROUNDS, HISTORY_SLOTS, RAISES_INDEX, HeadsUpPoker, history_slot
from headsup.enums import Action
from headsup.game import DEFAULT_GAME


def hero_cards(obs):
    """Card ids of the two hole cards in an observation row."""
    return [int(obs[2]) - 1, int(obs[5]) - 1]


def board_cards(obs):
    """Card ids of the dealt board cards in an observation row (3 / 4 / 5 or none)."""
    return [int(obs[6 + 3 * i + 2]) - 1 for i in range(5) if obs[6 + 3 * i + 2] > 0]


def actions_for_amount(engine, amount):
    """The engine actions that put ``amount`` chips in for the current player: check/call when it
    is the call amount, raise sizes whose amount matches, all-in when it is the whole stack (a
    raise that the raise cap converts into an all-in also puts the whole stack in - both
    candidates are returned, the raise first)."""
    to_call = engine.to_call
    stack = engine.stacks[engine.current]
    if amount == min(to_call, stack):
        return [int(Action.CHECK_CALL)]
    if amount == 0 and to_call > 0:  # a fold (recorded with size 0): only seen in terminal observations
        return [int(Action.FOLD)]
    out = []
    capped = engine.consecutive_raises + 1 >= engine.raise_cap
    for a in range(2, 2 + engine.game.num_raises):
        if engine.raise_amount(a) == amount or (capped and amount == stack and engine.all_in is not None):
            out.append(a)
    if amount == stack and engine.all_in is not None:
        out.append(engine.all_in)
    return out


def replay_from_obs(obs, game=DEFAULT_GAME, villain_cards=None, on_action=None):
    """Rebuild the hand's public state from ``obs`` (a row of the hero's observation).

    Returns ``(engine, hero_seat)``: an engine whose chips, board (known cards; unknown future
    cards are unused dummies), history and current player match the observed state; the hero
    holds its real cards, the opponent ``villain_cards`` (default: two unused dummies).
    ``on_action(engine, seat, action)`` is called before every replayed action (with the engine
    in the state the actor saw), which is how callers query a strategy at past decisions.
    """
    obs = np.asarray(obs, dtype=np.float32)
    hero = int(obs[22])
    mine = hero_cards(obs)
    board = board_cards(obs)
    used = set(mine) | set(board)
    if villain_cards is not None:
        used |= set(villain_cards)
    spare = [c for c in range(NUM_CARDS) if c not in used]
    villain = list(villain_cards) if villain_cards is not None else spare[:2]
    if villain_cards is None:
        spare = spare[2:]
    full_board = board + spare[: 5 - len(board)]
    deck = (mine + villain if hero == 0 else villain + mine) + full_board
    stage_now, raises_now = int(obs[21]), int(round(float(obs[RAISES_INDEX])))
    pot_now = int(round(float(obs[29]) * 1000))

    def replay(flip, notify):
        """Replay the recorded amounts; ``flip`` = (round, k) where the alternative candidate action
        is taken.  Returns the engine and the list of ambiguous steps."""
        engine = HeadsUpPoker(game=game)
        engine.reset(deck)
        ambiguous = []
        for r in range(HISTORY_ROUNDS):
            for k in range(HISTORY_SLOTS):
                size, occurred = float(obs[history_slot(r, k)]), float(obs[history_slot(r, k) + 1])
                if occurred == 0.0:
                    break
                if engine.done or int(engine.stage) != r:
                    raise ValueError(f"history has an action in round {r} but the replayed hand is at stage {int(engine.stage)}")
                amount = int(round(size * engine.pot))
                cands = actions_for_amount(engine, amount)
                if not cands:
                    raise ValueError(f"no action puts {amount} chips in at replay step (round {r}, k {k})")
                if len(cands) > 1:
                    ambiguous.append((r, k))
                action = cands[-1] if flip == (r, k) else cands[0]
                if notify and on_action is not None:
                    on_action(engine, engine.current, action)
                engine.step(action)
        return engine, ambiguous

    # a raise that the cap turned into an all-in and an explicit all-in put the same chips in and
    # only differ in the raise counter (obs[RAISES_INDEX]) - the observation tells which it was
    flip = None
    engine, ambiguous = replay(None, False)
    if not engine.done and engine.consecutive_raises != raises_now:
        current_street = [x for x in ambiguous if x[0] == stage_now]
        if current_street:
            flip = current_street[-1]
    engine, _ = replay(flip, True)
    if engine.pot != pot_now or int(engine.stage) != stage_now or (not engine.done and engine.current != hero):
        raise ValueError(f"replay mismatch: pot {engine.pot} vs {pot_now}, stage {int(engine.stage)} vs {stage_now}, current {engine.current} vs hero {hero}")
    return engine, hero
