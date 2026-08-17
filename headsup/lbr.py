"""Local Best Response (Lisý & Bowling 2017): a cheap, strong exploitability lower bound.

The LBR agent knows the opponent's strategy.  During a hand it keeps a Bayesian *range* over
the opponent's 1326 hole-card combinations, updated after every opponent action with the
probability the opponent's strategy assigns to that action for each combination (the strategy
is queried on the opponent's observation with the hand slots substituted).  At its own
decisions it evaluates every action with a one-street lookahead:

* fold: ``-own_bets``;
* check/call: both players check to showdown afterwards -> ``sum_h p(h) (2 eq(h) - 1) * pot_share``,
  where ``eq(h)`` is the hero's equity against combo ``h`` on the current board (exact over all
  runouts on the turn / river, Monte-Carlo before) and ``pot_share`` the chips at stake
  (``min`` of the two totals: an uncalled excess is returned);
* min-raise / all-in: the opponent folds with the probability its strategy gives at the
  resulting infoset (per combo), otherwise it calls (raises are treated as calls) and the hand
  is checked down;

and plays the best one (deterministic).  Values are chip results of the whole hand for the LBR
seat, so actions are directly comparable.  Chip outcomes over many hands (± standard error, and
optionally *duplicate* hands: the same deal replayed with the seats swapped, which cancels most
of the card luck) give the lower bound.

Any player spec can be the opponent (``cfr:...``, ``sdcfr:...``, ``onnx:...``, bots).  For
SD-CFR the queried model is a separate exact-mode instance whose per-table reach state is keyed
by ``table * 1326 + combo`` (see :class:`headsup.sdcfr.SDCFRPlayer`), so the queried strategy is
exactly the average strategy the sample-mode opponent realises.

    python -m headsup.lbr --policy sdcfr:runs/x/iterates.pt --hands 100000
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from headsup.cards import CARD_FEATURES, NUM_CARDS
from headsup.engine import HeadsUpPoker
from headsup.enums import Action
from headsup.game import action_label

NUM_COMBOS = 1326
COMBOS = np.array([(a, b) for a in range(NUM_CARDS) for b in range(a + 1, NUM_CARDS)], dtype=np.int64)  # index = combo_index
COMBO_FEATURES = CARD_FEATURES[COMBOS].reshape(NUM_COMBOS, 6)
_CARD_TO_COMBOS = [np.flatnonzero((COMBOS == c).any(axis=1)) for c in range(NUM_CARDS)]


def valid_combos(cards):
    """bool[1326]: combos that do not overlap ``cards``."""
    ok = np.ones(NUM_COMBOS, dtype=bool)
    for c in cards:
        ok[_CARD_TO_COMBOS[int(c)]] = False
    return ok


def transition_likelihood(engine, action, sigma):
    """P(the public transition produced by ``action`` | hand) for every combo, from a strategy
    ``sigma`` (1326, num_actions) at the engine's current decision: actions that lead to the same
    public state (e.g. a capped raise and an all-in) are indistinguishable, so their probabilities
    are summed.  The engine is left untouched."""
    outcomes = []
    legal = engine.legal_mask()
    for cand in range(engine.num_actions):
        if not legal[cand]:
            outcomes.append(None)
            continue
        c = engine.clone()
        c.step(cand)
        outcomes.append((c.done, c.folded, tuple(c.bets), tuple(c.stacks), int(c.stage)))
    c = engine.clone()
    c.step(int(action))
    taken = (c.done, c.folded, tuple(c.bets), tuple(c.stacks), int(c.stage))
    like = np.zeros(NUM_COMBOS)
    for cand, out in enumerate(outcomes):
        if out == taken:
            like += sigma[:, cand]
    return like


def substitute_hands(obs_row):
    """(1326, OBS_DIM): the observation ``obs_row`` with every possible hand in the hand slots."""
    rows = np.repeat(np.asarray(obs_row, dtype=np.float32)[None], NUM_COMBOS, axis=0)
    rows[:, :6] = COMBO_FEATURES
    return rows


def _model_for(spec, device, seed, model_iterates=0, game=None):
    """The player whose strategy LBR queries.

    SD-CFR: a separate exact-mode player over the same bank (optionally thinned to
    ``model_iterates`` representative iterates, see :meth:`headsup.sdcfr.IterateBank.thin`).
    """
    from headsup.players import make_player, parse_sdcfr_spec

    kind, _, arg = spec.partition(":")
    if kind.lower() == "sdcfr":
        from headsup.device import get_device
        from headsup.sdcfr import IterateBank, SDCFRPlayer

        path, _, gamma, iterations, thin = parse_sdcfr_spec(arg)
        bank = IterateBank.load(path, get_device(device), weight_power=gamma)
        if iterations:
            bank = bank.truncate(iterations)
        if thin:
            bank = bank.thin(thin)
        if model_iterates:
            bank = bank.thin(model_iterates)
        return SDCFRPlayer(bank, mode="exact", seed=seed)
    return make_player(spec, device=device, seed=seed, game=game)


class _Table:
    __slots__ = ("id", "engine", "seat", "range", "reward")

    def __init__(self, id, game):
        self.id = id
        self.engine = HeadsUpPoker(game=game)
        self.seat = 1  # flipped on the first reset
        self.range = None
        self.reward = None

    def start(self, seat, deck):
        self.seat = seat
        self.engine.reset(deck)
        self.range = valid_combos(self.engine.hands[seat]).astype(np.float64)
        self.range /= self.range.sum()
        self.reward = None

    @property
    def opp(self):
        return 1 - self.seat


class LocalBestResponse:
    def __init__(self, opponent_spec, num_tables=64, device=None, seed=0, mc_samples=200, max_exact=100,
                 duplicate=True, workers=None, engine_kwargs=None, model_iterates=0, opponent=None, model=None, game=None):
        from headsup.players import make_player

        from headsup.env import resolve_game

        self.spec = opponent_spec
        self.opponent = opponent if opponent is not None else make_player(opponent_spec, device=device, seed=seed, game=game)  # plays
        self.game = resolve_game(game, self.opponent, **(engine_kwargs or {}))
        self.model = model if model is not None else _model_for(opponent_spec, device, seed + 1, model_iterates, self.game)  # is queried
        self.model_observes = hasattr(self.model, "observe")
        self.duplicate = duplicate
        self.n = num_tables + (num_tables % 2 if duplicate else 0)
        self.rng = np.random.default_rng(seed)
        self.mc_samples, self.max_exact = mc_samples, max_exact
        self.tables = [_Table(i, self.game) for i in range(self.n)]
        from headsup import native

        self._cpp = native.module()
        self._pool = ThreadPoolExecutor(workers or max(1, min(32, (os.cpu_count() or 2) - 8)))
        self.num_actions = self.game.num_actions
        self.action_counts = np.zeros((4, self.num_actions), dtype=np.int64)  # per stage
        self.value_gap = []  # chosen value - value of check/call (diagnostic)

    # ------------------------------------------------------------------ hands
    def _new_deck(self):
        return self.rng.permutation(NUM_CARDS)[:9]

    def _start_table(self, t):
        if self.duplicate:
            partner = self.tables[t.id ^ 1]
            if t.id % 2 == 0:  # the even table draws the deck, seat 0; the odd one replays it in seat 1
                t.start(0, self._new_deck())
            else:
                t.start(1, list(partner.engine.hands[0]) + list(partner.engine.hands[1]) + list(partner.engine.board))
        else:
            t.start(1 - t.seat, self._new_deck())

    def _query(self, tables, obs_rows):
        """Opponent strategy for every combo at the given decision points: (len(tables), 1326, num_actions)."""
        rows = np.concatenate([substitute_hands(o) for o in obs_rows])
        ids = np.concatenate([t.id * NUM_COMBOS + np.arange(NUM_COMBOS) for t in tables])
        probs = self.model.probs(rows, ids)
        return np.asarray(probs, dtype=np.float64).reshape(len(tables), NUM_COMBOS, self.num_actions), rows, ids

    # ------------------------------------------------------------------ opponent moves
    def _opponent_step(self, tables):
        from headsup.env import _call_player

        real_obs = np.stack([t.engine.observation() for t in tables])
        actions = np.asarray(_call_player(self.opponent, real_obs, np.asarray([t.id for t in tables])))
        sigma, rows, ids = self._query(tables, real_obs)
        if self.model_observes:  # SD-CFR reach of every (table, combo) follows the action actually taken
            self.model.observe(rows, ids, np.repeat(actions, NUM_COMBOS))
        for t, a, sig in zip(tables, actions, sigma):
            e = t.engine
            like = transition_likelihood(e, int(a), sig)
            e.step(int(a))
            t.range *= like
            total = t.range.sum()
            if total <= 1e-12:  # the model gave this line probability ~0: fall back to the prior
                t.range = valid_combos(list(e.hands[t.seat]) + list(e.visible_board)).astype(np.float64)
                total = t.range.sum()
            t.range /= total

    # ------------------------------------------------------------------ LBR moves
    def _equities(self, tables):
        futs = [
            self._pool.submit(
                self._cpp.equity_vs_all, int(t.engine.hands[t.seat][0]), int(t.engine.hands[t.seat][1]),
                [int(c) for c in t.engine.visible_board], self.mc_samples, self.max_exact, int(self.rng.integers(2**63)),
            )
            for t in tables
        ]
        return [f.result().astype(np.float64) for f in futs]

    @staticmethod
    def _showdown_value(state, seat, p, eq):
        """Expected chips if the hand is checked down from ``state``: sum_h p(h) (2 eq(h) - 1) * stake."""
        stake = min(state.bets[seat], state.bets[1 - seat])
        return float(np.dot(p, 2.0 * eq - 1.0)) * stake

    def _lbr_step(self, tables):
        eqs = self._equities(tables)
        # hypothetical opponent decisions after each bet action
        bet_states, bet_query = [], []  # per table: {action: state}, list of (table index, action, obs)
        for i, t in enumerate(tables):
            e = t.engine
            states = {}
            legal = e.legal_mask()
            for a in range(2, self.num_actions):
                if not legal[a]:
                    continue
                c = e.clone()
                c.step(a)
                states[a] = c
                if not c.done:
                    bet_query.append((i, a, c.observation(t.opp)))
            bet_states.append(states)
        fold_probs = {}
        if bet_query:
            sig, _, _ = self._query([tables[i] for i, _, _ in bet_query], [o for _, _, o in bet_query])
            for (i, a, _), s in zip(bet_query, sig):
                fold_probs[(i, a)] = s[:, Action.FOLD]
        for i, t in enumerate(tables):
            e, seat, p, eq = t.engine, t.seat, t.range, eqs[i]
            p = np.where(eq >= 0, p, 0.0)  # blocked combos (new board cards) carry no mass
            p = p / p.sum() if p.sum() > 0 else valid_combos(list(e.hands[seat]) + list(e.visible_board)) / 1.0
            eq = np.where(eq >= 0, eq, 0.5)
            values = np.full(self.num_actions, -np.inf)
            if e.fold_allowed:
                values[Action.FOLD] = -e.bets[seat]
            c = e.clone()
            c.step(Action.CHECK_CALL)
            values[Action.CHECK_CALL] = self._showdown_value(c, seat, p, eq)
            for a, s in bet_states[i].items():
                if s.done:
                    values[a] = self._showdown_value(s, seat, p, eq)
                    continue
                pf = fold_probs[(i, a)]
                call_state = s.clone()
                call_state.step(Action.CHECK_CALL)
                stake = min(call_state.bets[seat], call_state.bets[1 - seat])
                values[a] = float(np.dot(p, pf * s.bets[1 - seat] + (1.0 - pf) * (2.0 * eq - 1.0) * stake))
            a = int(np.argmax(values))
            self.action_counts[int(e.stage), a] += 1
            self.value_gap.append(values[a] - values[Action.CHECK_CALL])
            e.step(a)

    # ------------------------------------------------------------------ main loop
    def play(self, hands, progress=True):
        """Play ``hands`` hands; returns per-hand LBR chip results (duplicate: per-pair means)."""
        from tqdm import tqdm

        results = []
        for t in self.tables:  # even tables draw the decks, so they start first
            if not self.duplicate or t.id % 2 == 0:
                self._start_table(t)
        if self.duplicate:
            for t in self.tables[1::2]:
                self._start_table(t)
        bar = tqdm(total=hands, disable=not progress, desc="lbr")
        while len(results) < hands:
            opp = [t for t in self.tables if not t.engine.done and t.engine.current == t.opp]
            if opp:
                self._opponent_step(opp)
            lbr = [t for t in self.tables if not t.engine.done and t.engine.current == t.seat]
            if lbr:
                self._lbr_step(lbr)
            for t in self.tables:
                if not t.engine.done or t.reward is not None:
                    continue
                t.reward = float(t.engine.rewards[t.seat])
                if not self.duplicate:
                    results.append(t.reward)
                    bar.update(1)
                    self._start_table(t)
                    continue
                partner = self.tables[t.id ^ 1]
                if partner.reward is not None:  # both tables of the pair are done
                    results.append(0.5 * (t.reward + partner.reward))
                    bar.update(1)
                    self._start_table(self.tables[t.id & ~1])
                    self._start_table(self.tables[t.id | 1])
        bar.close()
        return np.asarray(results[:hands], dtype=np.float64)

    def summary(self, results):
        m, se = float(results.mean()), float(results.std(ddof=1) / np.sqrt(len(results)))
        stages = ["preflop", "flop", "turn", "river"]
        counts = {s: {action_label(self.game, a): int(self.action_counts[i, a]) for a in range(self.num_actions)} for i, s in enumerate(stages)}
        return {
            "policy": self.spec, "hands": int(len(results)), "duplicate": self.duplicate,
            "lbr_chips_per_hand": m, "se": se, "mbb_per_hand": 500 * m, "mbb_se": 500 * se,
            "model_iterates": getattr(getattr(self.model, "bank", None), "T", None),
            "lbr_actions_by_stage": counts,
            "mean_value_gap_vs_call": float(np.mean(self.value_gap)) if self.value_gap else 0.0,
        }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="cfr", help="player spec of the strategy to exploit")
    p.add_argument("--hands", type=int, default=20_000, help="hands (duplicate: pairs of hands)")
    p.add_argument("--num-tables", type=int, default=64, help="tables played in lock-step")
    p.add_argument("--mc-samples", type=int, default=200, help="Monte-Carlo runouts for pre-flop / flop equities")
    p.add_argument("--max-exact", type=int, default=100, help="enumerate all runouts when there are at most this many (turn: 46; flop: 1081 -> Monte-Carlo)")
    p.add_argument("--no-duplicate", action="store_true", help="independent hands instead of duplicate pairs")
    p.add_argument("--workers", type=int, default=None, help="threads for the equity kernel")
    p.add_argument("--model-iterates", type=int, default=0,
                   help="SD-CFR: query a bank thinned to this many representative iterates (0 = all; the opponent still plays all)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write the summary to this file")
    args = p.parse_args(argv)

    lbr = LocalBestResponse(args.policy, num_tables=args.num_tables, device=args.device, seed=args.seed,
                            mc_samples=args.mc_samples, max_exact=args.max_exact, duplicate=not args.no_duplicate,
                            workers=args.workers, model_iterates=args.model_iterates)
    t0 = time.perf_counter()
    results = lbr.play(args.hands)
    summary = lbr.summary(results)
    summary["seconds"] = time.perf_counter() - t0
    print(f"LBR vs {args.policy}: {summary['lbr_chips_per_hand']:+.3f} ± {summary['se']:.3f} chips/hand "
          f"({summary['mbb_per_hand']:+.0f} ± {summary['mbb_se']:.0f} mbb/g) over {summary['hands']:,} "
          f"{'duplicate pairs' if not args.no_duplicate else 'hands'} in {summary['seconds']:.0f}s")
    print("LBR actions by stage:", json.dumps(summary["lbr_actions_by_stage"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
