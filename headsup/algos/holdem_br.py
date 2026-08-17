"""Exploitability of hold'em strategies: exact best response over all 1326 hands (vector form),
Monte-Carlo over the boards.

For FHP / HULH (and, in principle, the no-limit abstraction) the public betting tree of every
street is small, but the boards are many (22 100 flops); the papers' exact numbers use
domain-specific solvers.  We compute the best response of each player *exactly* over hands and
betting sequences for a set of sampled boards (each sample is one full board that the hand
follows through the streets), which is an unbiased Monte-Carlo estimate of the value of a best
response against the strategy - and, as with any best response, a *lower* bound on the true
exploitability up to the sampling of boards (the response may adapt to the sample; the bias
vanishes with the number of boards).  Strategies are queried at every public node for all 1326
hands at once (``player.probs`` on hand-substituted observations), terminal values use the
strength-sorted cumulative sums with blocker corrections of :mod:`headsup.search`.

    python -m headsup.algos.holdem_br --policy cfr:runs/fhp/policy.pth --game fhp --boards 200
"""

import argparse
import json
import time

import numpy as np

from headsup import native
from headsup.cards import NUM_CARDS, hand_strength
from headsup.engine import BOARD_CARDS_BY_STAGE, HeadsUpPoker
from headsup.game import FHP, GameConfig
from headsup.lbr import COMBOS, NUM_COMBOS, substitute_hands, valid_combos
from headsup.search import _opponent_mass, _showdown_values


class _Node:
    __slots__ = ("engine", "kind", "player", "folder", "stake", "children", "legal", "twins", "stage")

    def __init__(self, engine):
        self.engine = engine
        self.stage = int(engine.stage)
        if engine.done:
            self.kind = 1 if engine.folded >= 0 else 2
            self.player = -1
            self.folder = engine.folded
            self.stake = engine.bets[engine.folded] if engine.folded >= 0 else min(engine.bets)
            self.children = {}
            self.legal = self.twins = None
        else:
            self.kind = 0
            self.player = engine.current
            self.folder, self.stake = -1, 0
            self.legal, self.twins = engine.legal_mask_and_twins()
            self.children = {}


def build_street_tree(engine):
    """Public tree of the current street: dict id -> _Node; children of decision nodes per legal
    action, street-end leaves are decision nodes of the *next* street (kind 3)."""
    nodes = [_Node(engine)]
    i = 0
    while i < len(nodes):
        nd = nodes[i]
        if nd.kind == 0 and nd.stage == nodes[0].stage:
            for a in np.flatnonzero(nd.legal):
                c = nd.engine.clone()
                c.step(int(a))
                child = _Node(c)
                if child.kind == 0 and child.stage != nodes[0].stage:
                    child.kind = 3  # street-end leaf
                nd.children[int(a)] = len(nodes)
                nodes.append(child)
        i += 1
    return nodes


def chance_factor(revealed_before, revealed_after):
    """P(the board cards revealed between the two counts are compatible with a fixed pair of hands)
    when the board is drawn uniformly: the same for every hand pair (4 blocked cards), which is
    what makes ``sum over sampled boards / (N * factor)`` an unbiased, exactly zero-sum estimate."""
    f = 1.0
    for i in range(revealed_before, revealed_after):
        f *= (NUM_CARDS - 4 - i) / (NUM_CARDS - i)
    return f


class _NativeShowdown:
    def __init__(self, board):
        self.table = native.module().BoardTable(board, len(board))

    def __call__(self, reach):
        return self.table.showdown_values(reach)


class _NumpyShowdown:
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, reach):
        return _showdown_values(reach, self.strength)


class HoldemBestResponse:
    def __init__(self, player, game=FHP, boards=200, seed=0, batch=8192):
        self.player, self.game = player, game
        self.rng = np.random.default_rng(seed)
        self.n_boards = boards
        self.batch = batch
        self.probs_cache = {}
        self.calls = 0

    # -- strategy queries -------------------------------------------------------------------
    def _strategy(self, node, seat):
        """(1326, A) strategy of ``seat`` at the public node (its observation with every hand)."""
        e = node.engine
        key = (seat, tuple(e.visible_board), tuple(e.bets), tuple(e.stage_bets), int(e.stage), e.consecutive_raises,
               tuple(tuple(x) for x in e.history_size), tuple(e.history_n))
        cached = self.probs_cache.get(key)
        if cached is not None:
            return cached
        rows = substitute_hands(e.observation(seat))
        probs = np.asarray(self.player.probs(rows), dtype=np.float64)
        self.calls += 1
        self.probs_cache[key] = probs
        return probs

    # -- values -------------------------------------------------------------------------------
    def _values(self, nodes, node_id, p, reach_q, boards, strength, best_response):
        """Counterfactual value vector of player p's hands at ``node_id`` given the opponent's
        reach vector; ``boards`` = the sampled boards this subtree may see (list of card tuples),
        ``strength`` = per-board strength arrays for the showdown."""
        nd = nodes[node_id]
        if nd.kind == 1:
            sign = 1.0 if nd.folder != p else -1.0
            return sign * nd.stake * _opponent_mass(reach_q)
        n_final = BOARD_CARDS_BY_STAGE[self.game.num_rounds - 1]
        revealed = len(nd.engine.visible_board)
        if nd.kind == 2:
            # showdown on the boards' final cards (the unrevealed ones are dealt): sum over the sampled
            # boards, hands overlapping a board contribute nothing there, normalised by N * P(compatible)
            out = np.zeros(NUM_COMBOS)
            for b, s in zip(boards, strength):
                ok = valid_combos(b[:n_final])
                out += nd.stake * s(np.where(ok, reach_q, 0.0)) * ok
            return out / (len(boards) * chance_factor(revealed, n_final))
        if nd.kind == 3:
            # street end: continue on each sampled board with its next street's cards
            out = np.zeros(NUM_COMBOS)
            nxt = None
            for b, s in zip(boards, strength):
                sub = self._street_from(nd.engine, b)
                nxt = BOARD_CARDS_BY_STAGE[int(sub[0].engine.stage)]
                ok = valid_combos(b[:nxt])
                out += self._values(sub, 0, p, np.where(ok, reach_q, 0.0), [b], [s], best_response) * ok
            return out / (len(boards) * chance_factor(revealed, nxt))
        acts = [a for a in range(self.game.num_actions) if nd.legal[a]]
        if nd.player == p:
            child_values = np.stack([self._values(nodes, nd.children[a], p, reach_q, boards, strength, best_response) for a in acts])
            if best_response:
                return child_values.max(axis=0)
            sig = self._strategy(nd, p)
            return (sig[:, acts].T * child_values).sum(axis=0)
        sig = self._strategy(nd, nd.player)
        return sum(self._values(nodes, nd.children[a], p, reach_q * sig[:, a], boards, strength, best_response) for a in acts)

    def _street_from(self, engine, board):
        """Public tree of the next street after ``engine``'s street-end state on ``board``."""
        e = engine.clone()
        e.board = tuple(board)  # the sampled board (the engine's dummy cards are replaced)
        spare = [c for c in range(NUM_CARDS) if c not in board][:4]  # dummy hands off the board (showdowns evaluate them)
        e.hands = ((spare[0], spare[1]), (spare[2], spare[3]))
        return build_street_tree(e)

    def _strengths(self, boards):
        """Per board: the showdown-payoff evaluator (the C++ table when the extension is built,
        otherwise the numpy strength-sorted cumulative sums of headsup.search)."""
        n_final = BOARD_CARDS_BY_STAGE[self.game.num_rounds - 1]
        if native.available():
            return [_NativeShowdown(list(b[:n_final])) for b in boards]
        out = []
        for b in boards:
            s = np.full(NUM_COMBOS, np.inf)
            final = list(b[:n_final])
            for h in np.flatnonzero(valid_combos(final)):
                s[h] = hand_strength([int(COMBOS[h, 0]), int(COMBOS[h, 1])], final)
            out.append(_NumpyShowdown(s))
        return out

    def evaluate_from(self, engine, ranges, boards):
        """Best responses from a given public state with given ranges over the given full boards
        (a list of 5-card tuples whose known prefix must match the engine's board)."""
        root = build_street_tree(engine)
        strength = self._strengths(boards)
        r = [np.asarray(x, dtype=np.float64) for x in ranges]
        joint = float(_opponent_mass(r[1]) @ r[0])
        out = {}
        for p in (0, 1):
            br = float(r[p] @ self._values(root, 0, p, r[1 - p], boards, strength, True)) / joint
            v = float(r[p] @ self._values(root, 0, p, r[1 - p], boards, strength, False)) / joint
            out[p] = (br, v)
        expl = 0.5 * (out[0][0] + out[1][0])
        return {"exploitability_chips": expl, "exploitability_mbb": 1000.0 * expl / self.game.big_blind,
                "br_values": [out[0][0], out[1][0]], "values": [out[0][1], out[1][1]], "boards": len(boards), "strategy_queries": self.calls}

    def run(self):
        e = HeadsUpPoker(rng=self.rng, game=self.game)
        e.reset()
        boards = [tuple(int(c) for c in self.rng.permutation(NUM_CARDS)[:5]) for _ in range(self.n_boards)]
        r = np.ones(NUM_COMBOS)  # uniform ranges (the deal)
        return self.evaluate_from(e, [r, r], boards)


def main(argv=None):
    from headsup.games.holdem import make_holdem
    from headsup.players import make_player

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="stateless player spec (cfr:..., iterate:...)")
    p.add_argument("--game", default="fhp", choices=["fhp", "hulh", "nlhe"])
    p.add_argument("--boards", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    game = make_holdem(args.game)
    player = make_player(args.policy, device=args.device, seed=args.seed, game=game)
    if getattr(player, "game", None) is not None:  # network / tabular players carry their action tree
        if player.game.tree_dict() != game.tree_dict():
            print(f"(using the player's action tree {player.game.tree_dict()} instead of --game {args.game})")
        game = player.game
    t0 = time.perf_counter()
    res = HoldemBestResponse(player, game, boards=args.boards, seed=args.seed).run()
    res["seconds"] = time.perf_counter() - t0
    print(f"{args.policy} on {args.game}: exploitability {res['exploitability_mbb']:.1f} mbb/g "
          f"({res['exploitability_chips']:.2f} chips; BR values {res['br_values'][0]:+.2f} / {res['br_values'][1]:+.2f}, "
          f"profile values {res['values'][0]:+.2f} / {res['values'][1]:+.2f}) over {args.boards} boards in {res['seconds']:.0f}s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
