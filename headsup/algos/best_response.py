"""Exact best response and exploitability for enumerable games (Kuhn, Leduc, ...).

``exploitability(game, policy)`` returns ``(exploitability, br_values)`` where ``br_values[p]`` is
the value player p obtains by best-responding to ``policy`` (the opponent playing ``policy``), and
exploitability is their average - the standard measure the papers report (0 at a Nash
equilibrium).  ``policy(state) -> probs`` gives the strategy at a decision state (any callable:
tabular tables, a network wrapper, ...).

Best responses are computed per information set: q(I, a) = sum over the states s of I of the
opponent's reach probability of s times the value of a's child under the best response below;
states in one infoset share the argmax (perfect recall makes the recursion well founded).
"""

from collections import defaultdict

import numpy as np



class TabularPolicy:
    """policy(state) from a dict info_key -> probs (uniform over legal actions when missing)."""

    def __init__(self, game, table=None):
        self.game = game
        self.table = table if table is not None else {}

    def __call__(self, state):
        p = self.table.get(state.info_key(state.current_player))
        if p is None:
            p = np.zeros(self.game.num_actions)
            p[state.legal_actions()] = 1.0
            return p / p.sum()
        return p


def best_response(game, policy, br_player):
    """Value of ``br_player``'s best response to ``policy``; also returns the response as a table."""
    # 1. collect every state of br_player's infosets with the opponent's (and chance's) reach
    infosets = defaultdict(list)  # key -> [(state, reach)]

    def collect(state, reach):
        if state.is_terminal():
            return
        if state.is_chance():
            for a, p in state.chance_outcomes():
                collect(state.child(a), reach * p)
            return
        if state.current_player == br_player:
            infosets[state.info_key(br_player)].append((state, reach))
            for a in state.legal_actions():
                collect(state.child(a), reach)
        else:
            probs = policy(state)
            for a in state.legal_actions():
                if probs[a] > 0:
                    collect(state.child(a), reach * probs[a])

    collect(game.new_initial_state(), 1.0)
    decision = {}

    def value(state):
        """br_player's value of the subtree (chance / opponent weighted; own decisions per infoset)."""
        if state.is_terminal():
            return state.returns()[br_player]
        if state.is_chance():
            return sum(p * value(state.child(a)) for a, p in state.chance_outcomes())
        if state.current_player != br_player:
            probs = policy(state)
            return sum(probs[a] * value(state.child(a)) for a in state.legal_actions() if probs[a] > 0)
        key = state.info_key(br_player)
        if key not in decision:
            q = defaultdict(float)
            for s, reach in infosets[key]:
                for a in s.legal_actions():
                    q[a] += reach * value(s.child(a))
            decision[key] = max(q, key=q.get)
        return value(state.child(decision[key]))

    root_value = value(game.new_initial_state())
    table = {}
    for key, a in decision.items():
        p = np.zeros(game.num_actions)
        p[a] = 1.0
        table[key] = p
    return root_value, table


def expected_value(game, policy, policy2=None):
    """Player 0's expected return when player 0 plays ``policy`` and player 1 ``policy2`` (default: the same)."""
    policy2 = policy2 or policy

    def value(state):
        if state.is_terminal():
            return state.returns()[0]
        if state.is_chance():
            return sum(p * value(state.child(a)) for a, p in state.chance_outcomes())
        probs = (policy if state.current_player == 0 else policy2)(state)
        return sum(probs[a] * value(state.child(a)) for a in state.legal_actions() if probs[a] > 0)

    return value(game.new_initial_state())


def exploitability(game, policy):
    br = [best_response(game, policy, p)[0] for p in range(2)]
    return 0.5 * (br[0] + br[1]), br
