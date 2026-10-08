"""Game session behind the web UI: one human seat vs a bot, with history, advisor and stats.

The session drives :class:`headsup.engine.HeadsUpPoker` directly (rather than through
``SingleAgentEnv``) so every bot action can be recorded with its amount for the hand log.
"""

import os
import threading
import time

import numpy as np

from headsup.cards import card_to_str, describe_hand, hand_strength, showdown_equity
from headsup.engine import HeadsUpPoker
from headsup.enums import Action, Stage
from headsup.game import DEFAULT_GAME, action_label
from headsup.paths import DEFAULT_POLICY_PATH
from headsup.players import make_player

BOT_LABELS = {
    "cfr": "DeepCFR", "sdcfr": "SD-CFR", "onnx": "PPO exploiter", "random": "Random bot", "call": "Calling station",
    "allin": "Maniac", "raise": "Always-raise bot", "tab": "Tabular blueprint", "pluribus": "Pluribus-style (blueprint + search)",
    "search": "Search",
}


def bot_label(spec):
    kind, _, arg = str(spec).partition(":")
    if kind.startswith("pluribus"):
        return BOT_LABELS["pluribus"]
    if kind in BOT_LABELS and not arg:
        return BOT_LABELS[kind]
    name = os.path.basename(arg or spec)
    return f"{BOT_LABELS.get(kind, kind)} · {name}" if kind in BOT_LABELS else name
SUIT_GLYPH = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RANK_LABEL = {"T": "10"}


ACTION_ALIASES = {
    "fold": 0, "check": 1, "call": 1, "check_call": 1, "raise": 2, "bet": 2, "allin": "allin", "all_in": "allin", "all-in": "allin",
}


def parse_action(value, game=DEFAULT_GAME):
    """Action index from an int, an alias (fold / check / call / raise / allin) or a game label
    (``action_label``: fold, call, raise_min, raise_0.5p, ..., allin)."""
    if isinstance(value, (int, np.integer)) or str(value).lstrip("-").isdigit():
        a = int(value)
        if not 0 <= a < game.num_actions:
            raise ValueError(f"unknown action {value!r}")
        return a
    key = str(value).strip().lower()
    labels = [action_label(game, a) for a in range(game.num_actions)]
    if key in labels:
        return labels.index(key)
    if key not in ACTION_ALIASES:
        raise ValueError(f"unknown action {value!r}")
    a = ACTION_ALIASES[key]
    if a == "allin":
        if game.all_in_action is None:
            raise ValueError("this game has no all-in action")
        return game.all_in_action
    return a


def size_label(size):
    if size == "min":
        return "min"
    return {0.25: "¼ pot", 0.5: "½ pot", 0.75: "¾ pot", 1.0: "pot", 2.0: "2× pot"}.get(float(size), f"{float(size):g}× pot")


def player_spec(value):
    """Accept player specs ('cfr', 'onnx:path', 'sdcfr:path', 'call', ...) as well as bare model paths."""
    value = str(value).strip()
    if ":" in value or value in ("cfr", "onnx", "random", "call", "allin", "raise", "tab") or value.startswith("pluribus"):
        return value
    if value.endswith(".onnx"):
        return f"onnx:{value}"
    if value.endswith("iterates.pt"):
        return f"sdcfr:{value}"
    return f"cfr:{value}"


def card_json(card):
    s = card_to_str(card)
    return {"rank": RANK_LABEL.get(s[0], s[0]), "suit": SUIT_GLYPH[s[1]], "red": s[1] in "hd", "id": int(card), "str": s}


def equity(hand, board, rng, samples=3000):
    """Monte-Carlo win/tie probability of ``hand`` vs a random hand, running out ``board``."""
    try:
        from headsup import native

        cpp = native.module()
        ev = lambda h, b: cpp.eval7(list(h) + list(b))
    except ImportError:
        ev = hand_strength
    used = set(hand) | set(board)
    remaining = np.array([c for c in range(52) if c not in used])
    need = 2 + (5 - len(board))
    win = tie = 0
    for _ in range(samples):
        draw = rng.choice(remaining, size=need, replace=False)
        full_board = list(board) + [int(c) for c in draw[2:]]
        mine = ev(hand, full_board)
        theirs = ev((int(draw[0]), int(draw[1])), full_board)
        if mine < theirs:
            win += 1
        elif mine == theirs:
            tie += 1
    return win / samples, tie / samples


class GameSession:
    def __init__(
        self,
        opponent="cfr",
        advisor=DEFAULT_POLICY_PATH,
        stack_size=None,
        small_blind=None,
        big_blind=None,
        raise_cap=None,
        seed=None,
        deterministic=False,
        device=None,
    ):
        self.lock = threading.RLock()
        if advisor and os.path.abspath(str(advisor)) == os.path.abspath(DEFAULT_POLICY_PATH):
            advisor = "cfr"
        self.settings = dict(opponent=opponent, advisor=advisor, seed=seed, deterministic=bool(deterministic))
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.opponent = make_player(player_spec(opponent), device=device, deterministic=deterministic, seed=int(self.rng.integers(2**31)))
        self.opponent_name = opponent
        # the table plays the bot's game: its action tree (bet sizes) and, unless the settings say otherwise, the
        # stacks / blinds / raise cap it was trained with.  Bots on a fixed public tree (tabular blueprint, search)
        # only know their own game: for them the settings are ignored.
        bot_game = getattr(self.opponent, "game", None) or DEFAULT_GAME
        if bot_game.limit is not None:
            raise ValueError("limit games are not supported by the browser table (its controls are no-limit: "
                             f"bet sizes and all-in); {opponent!r} plays {bot_game.tree_dict()}")
        asked = dict(stack_size=stack_size, small_blind=small_blind, big_blind=big_blind, raise_cap=raise_cap)
        if type(self.opponent).__name__ in ("TabularPlayer", "SearchPlayer"):
            asked = dict.fromkeys(asked)
        table = {k: int(v) if v is not None else getattr(bot_game, k) for k, v in asked.items()}
        self.settings.update(table)
        self.game = bot_game.with_(**table)
        changed = [f"{k.replace('_', ' ')} {getattr(bot_game, k)}" for k in table if getattr(self.opponent, "game", None) is not None
                   and table[k] != getattr(bot_game, k)]
        self.warning = f"the bot was trained with {', '.join(changed)}: at this table it is off its training game" if changed else None
        self.engine = HeadsUpPoker(game=self.game, rng=np.random.default_rng(self.rng.integers(2**63)))
        self.action_names = [action_label(self.game, a) for a in range(self.game.num_actions)]
        self.action_labels = ["Fold", "Check/Call"] + [
            "Raise" if s == "min" and len(self.game.bet_sizes) == 1 else f"Raise {size_label(s)}" for s in self.game.bet_sizes
        ] + ["All-in"]
        self.advisor = None
        self.advisor_error = None
        if advisor:
            try:
                self.advisor = make_player(player_spec(advisor), device=device, seed=int(self.rng.integers(2**31)), game=self.game)
                if getattr(self.advisor, "game", self.game).num_actions != self.game.num_actions:
                    raise ValueError("advisor plays a different action set than the bot")
            except Exception as exc:  # advisor is optional
                self.advisor = None
                self.advisor_error = f"{type(exc).__name__}: {exc}"
        self.me = 1  # flipped in new_hand -> human is dealer first
        self.results = []  # per-hand rewards for the human
        self.ev_results = []  # the same with all-in hands at their expectation over the runouts (all-in EV)
        # ... and with showdowns at their expectation over the hands the bot plays this way (see _range_ev)
        self.adj_results = []
        self.runout_samples = 0  # 0: exact expectations over the runouts (pre-flop all-ins: ~0.1 s); > 0: that many sampled
        self._bot_range = None  # P(the bot's actions of this hand | each of the 1326 hands), or None when unknown
        self.hand_records = []  # compact summaries of finished hands
        self.log = []  # action log of the current hand
        self.hand_over = False
        self.result = None
        self.bot_last = None  # {"action":..., "probs": [...]}
        self._equity_cache = {}
        self.new_hand()

    # ------------------------------------------------------------------ helpers
    @property
    def opp(self):
        return 1 - self.me

    def _pos(self, seat):
        return "dealer" if seat == self.engine.dealer else "big blind"

    def _describe_action(self, seat, action, before):
        """Human readable action label given the engine state *before* the step."""
        e = self.engine
        o = 1 - seat
        to_call = before["stage_bets"][o] - before["stage_bets"][seat]
        stack = before["stacks"][seat]
        action = int(action)
        if action == Action.FOLD and to_call > 0:
            return "folds", 0
        if action in (Action.FOLD, Action.CHECK_CALL):  # a fold with nothing to call is executed as a check
            amt = min(to_call, stack)
            return ("checks", 0) if amt == 0 else (f"calls {amt}", amt)
        if self.game.is_raise(action):
            verb = "bets" if to_call == 0 else "raises"
            amt = self.game.raise_amount(action, to_call, before["pot"], stack)
            if before["consecutive_raises"] + 1 >= e.game.cap(int(before["stage"])) or amt >= stack:
                return f"{verb} all-in {stack}", stack
            return (f"bets {amt}", amt) if to_call == 0 else (f"raises to {before['stage_bets'][seat] + amt}", amt)
        return f"all-in {stack}", stack

    def _snapshot(self):
        e = self.engine
        return dict(stage_bets=list(e.stage_bets), stacks=list(e.stacks), consecutive_raises=e.consecutive_raises, pot=e.pot, stage=int(e.stage))

    def _record(self, seat, action, before):
        text, amount = self._describe_action(seat, action, before)
        self.log.append(
            {
                "seat": "you" if seat == self.me else "bot",
                "stage": Stage(self.engine.stage).name.lower() if not self.engine.done else "end",
                "action": self.action_names[int(action)],
                "text": text,
                "amount": amount,
                "t": time.time(),
            }
        )

    def _step(self, seat, action):
        before = self._snapshot()
        stage_before = self.engine.stage
        self._record(seat, action, before)
        _, rewards, done, _ = self.engine.step(int(action))
        if not done and self.engine.stage != stage_before:
            self.log.append({"seat": "table", "stage": Stage(self.engine.stage).name.lower(), "text": "board", "action": "board", "amount": 0, "t": time.time()})
        if done:
            self._finish()

    @property
    def bot_to_act(self):
        return (not self.hand_over) and self.engine.current == self.opp

    def bot_step(self):
        """Let the bot make exactly one decision (the client paces these for a natural feel)."""
        if not self.bot_to_act:
            return self.state()
        e = self.engine
        obs = e.observation()[None]
        action = int(self.opponent(obs)[0])
        probs = getattr(self.opponent, "last_probs", None)
        self.bot_last = {
            "action": self.action_names[int(action)],
            "text": self._describe_action(self.opp, action, self._snapshot())[0],
            "probs": [float(p) for p in np.asarray(probs).reshape(-1)] if probs is not None else None,
            "hand": e.hands_played,
            "n": len(self.log),
        }
        self._update_bot_range(obs, action)
        self._step(self.opp, action)
        return self.state()

    def _update_bot_range(self, obs, action):
        """Multiply the bot's range by the probability that it takes ``action`` with each of the 1326 hands (its
        strategy at this public state for every hand: asked from a search player's round solve, or by substituting
        the hands into the observation).  Bots whose strategy depends on per-hand state cannot say: no range."""
        if self._bot_range is None:
            return
        from headsup.lbr import substitute_hands, transition_likelihood

        opp = self.opponent
        try:
            if getattr(opp, "answers_all_hands", False):
                sigma = np.asarray(opp.all_hands_probs(obs, np.array([0]))[0], dtype=np.float64)
            elif getattr(opp, "wants_ids", False) or not hasattr(opp, "probs"):
                raise ValueError("per-hand state")
            else:
                sigma = np.asarray(opp.probs(substitute_hands(obs[0])), dtype=np.float64)
            if getattr(opp, "deterministic", False):  # it plays the most likely action of each hand
                sigma = np.eye(sigma.shape[1])[sigma.argmax(axis=1)]
            self._bot_range = self._bot_range * transition_likelihood(self.engine, action, sigma)
        except Exception:  # the adjustment is optional: never let it break a hand
            self._bot_range = None

    def _advance_bot(self):
        """Advance the bot until it is the human's turn or the hand is over (used by autoplay/tests)."""
        while self.bot_to_act:
            self.bot_step()

    def _finish(self):
        e = self.engine
        reward = int(e.rewards[self.me])
        self.hand_over = True
        self.results.append(reward)
        showdown = e.folded < 0
        main = f"You win {reward}" if reward > 0 else f"You lose {-reward}" if reward < 0 else "Split pot"
        info = {"reward": reward, "showdown": showdown, "pot": e.pot, "main": main}
        if showdown:
            mine, theirs = describe_hand(e.hands[self.me], e.board), describe_hand(e.hands[self.opp], e.board)
            info.update(my_class=mine, bot_class=theirs)
            if reward > 0:
                info["detail"] = f"your {mine} beats {theirs}"
            elif reward < 0:
                info["detail"] = f"{theirs} beats your {mine}"
            else:
                info["detail"] = f"both {mine}"
        elif e.folded == self.opp:
            info["detail"] = "the bot folded"
        else:
            info["detail"] = "you folded"
        info["text"] = f"{main} — {info['detail']}"
        info["allin"] = self._allin_ev(reward)
        ev = info["allin"]["ev"] if info["allin"] else float(reward)
        self.ev_results.append(ev)
        info["vs_range"] = self._range_ev(reward) if showdown else None
        adj = info["vs_range"]["ev"] if info["vs_range"] else ev
        self.adj_results.append(adj)
        self.result = info
        self.hand_records.append(
            {
                "hand": e.hands_played,
                "reward": reward,
                "showdown": showdown,
                "you": [card_json(c) for c in e.hands[self.me]],
                "bot": [card_json(c) for c in e.hands[self.opp]] if showdown else None,
                "board": [card_json(c) for c in (e.board if showdown else e.visible_board)],
                "position": self._pos(self.me),
                "summary": info["text"],
                "ev": ev,
                "adj": adj,
            }
        )
        self.hand_records = self.hand_records[-200:]

    def _range_ev(self, reward):
        """A showdown valued against every hand the bot plays this way: the expectation of the result over the bot's
        hands, weighted by the probability that it takes its actions of this hand with each (and over the cards to
        come when the betting closed early).  Given what both players could see, which of those hands it held is
        luck - the average has the same mean over many hands and far less variance.  None when the bot's strategy
        for other hands is unknown or the C++ kernels are missing."""
        from headsup import native
        from headsup.lbr import valid_combos

        e = self.engine
        if self._bot_range is None or not native.available() or e.showdown_stage < 0:
            return None
        known = list(e.board[: (0, 3, 4, 5)[e.showdown_stage]])
        final = (0, 3, 4, 5)[e.num_rounds - 1]
        mine = list(e.hands[self.me])
        eq = np.asarray(native.module().equity_vs_all(mine[0], mine[1], known, self.runout_samples or 2000, 1100,
                                                      int(self.rng.integers(2**31)), final), dtype=np.float64)
        w = self._bot_range * valid_combos(mine + known) * (eq >= 0)
        if not w.sum() > 0:
            return None
        w = w / w.sum()
        equity = float(w @ np.where(eq >= 0, eq, 0.0))
        ev = float(min(e.bets) * (2.0 * equity - 1.0))
        return {"equity": equity, "ev": ev, "luck": reward - ev,
                "text": f"against all the hands the bot plays this way you have {equity:.0%}: worth {ev:+.1f} on average"}

    def _allin_ev(self, reward):
        """For a hand that went all-in before the last card: your equity when the money went in, what the hand was
        worth on average over the cards to come, and the part of the result that was the runout's luck."""
        e = self.engine
        if e.folded >= 0 or e.showdown_stage < 0 or e.showdown_stage >= e.num_rounds - 1:
            return None
        known = e.board[: (0, 3, 4, 5)[e.showdown_stage]]
        win, tie = showdown_equity(e.hands[self.me], e.hands[self.opp], known, len(e.board[: (0, 3, 4, 5)[e.num_rounds - 1]]),
                                   self.runout_samples, int(self.rng.integers(2**31)))
        equity, ev = win + tie / 2, float(e.allin_ev(self.runout_samples, int(self.rng.integers(2**31)))[self.me])
        street = ("pre-flop", "flop", "turn")[e.showdown_stage]
        return {"street": street, "equity": equity, "ev": ev, "luck": reward - ev,
                "text": f"all-in {'on the ' if e.showdown_stage else ''}{street} with {equity:.0%} equity: worth {ev:+.1f} on average, "
                        f"the runout gave you {reward - ev:+.1f}"}

    # ------------------------------------------------------------------ actions
    def new_hand(self):
        self.me ^= 1
        self.engine.reset()
        self.hand_over = False
        self.result = None
        self._bot_range = np.ones(1326)
        self.log = []
        self.bot_last = None
        self._equity_cache = {}
        return self.state()

    def act(self, action):
        if self.hand_over:
            raise ValueError("hand is over — start the next hand")
        action = parse_action(action, self.game)
        if self.engine.current != self.me:
            raise ValueError("not your turn")
        obs = self.engine.observation(self.me)[None]
        self._step(self.me, action)
        if self.advisor is not None and hasattr(self.advisor, "observe"):
            try:  # SD-CFR exact mode tracks the human's own reach within the hand
                self.advisor.probs(obs, [0])
                self.advisor.observe(obs, [0], [int(action)])
            except Exception:
                pass
        return self.state()

    def next_hand(self):
        """Start the next hand; a no-op (returns the state) while a hand is still running."""
        if not self.hand_over:
            return self.state()
        return self.new_hand()

    def agent_step(self, deterministic=False):
        """Let the advisor act for the human (autoplay)."""
        if self.hand_over:
            return self.new_hand()
        if self.bot_to_act:
            return self.bot_step()
        if self.advisor is None:
            raise ValueError("no advisor policy loaded")
        probs = self.advisor.probs(self.engine.observation(self.me)[None])[0]
        action = int(np.argmax(probs)) if deterministic else int(self.rng.choice(len(probs), p=probs / probs.sum()))
        return self.act(action)

    # ------------------------------------------------------------------ views
    def legal_actions(self):
        e = self.engine
        if self.hand_over or e.current != self.me:
            return []
        me, opp = self.me, self.opp
        to_call = e.stage_bets[opp] - e.stage_bets[me]
        stack = e.stacks[me]
        legal = e.legal_mask()
        out = [{"action": "fold", "label": "Fold", "key": "F"}] if legal[Action.FOLD] else []
        out.append({"action": "call", "label": "Check" if to_call == 0 else f"Call {min(to_call, stack)}", "key": "C"})
        verb = "Bet" if to_call == 0 else "Raise"
        multi = len(self.game.bet_sizes) > 1
        for k, size in enumerate(self.game.bet_sizes):
            a = 2 + k
            if not legal[a]:
                continue
            amt = e.raise_amount(a)
            key = str(k + 1) if multi else "R"
            suffix = f" ({size_label(size)})" if multi else ""
            if e.consecutive_raises + 1 >= e.game.cap(int(e.stage)) or amt >= stack:
                out.append({"action": self.action_names[a], "label": f"{verb} all-in {stack}{suffix}", "key": key, "allin": True})
            elif to_call == 0:
                out.append({"action": self.action_names[a], "label": f"Bet {amt}{suffix}", "key": key})
            else:
                out.append({"action": self.action_names[a], "label": f"Raise to {e.stage_bets[me] + amt}{suffix}", "key": key})
        out.append({"action": "allin", "label": f"All-in {stack}", "key": "A"})
        return out

    def stats(self):
        r = np.asarray(self.results, dtype=np.float64)
        n = len(r)
        cum = np.cumsum(r).tolist() if n else []
        return {
            "hands": n,
            "total": float(r.sum()) if n else 0.0,
            "avg": float(r.mean()) if n else 0.0,
            "mbb": float(r.mean() * 1000 / self.engine.big_blind) if n else 0.0,
            "se_mbb": float(r.std(ddof=1) / np.sqrt(n) * 1000 / self.engine.big_blind) if n > 1 else 0.0,
            "won": int((r > 0).sum()) if n else 0,
            "lost": int((r < 0).sum()) if n else 0,
            "tied": int((r == 0).sum()) if n else 0,
            "cumulative": cum[-400:],
            **self._ev_stats(r),
        }

    def _ev_stats(self, r):
        """The session with all-in hands at their expectation: the same quantity with less luck in it."""
        ev = np.asarray(self.ev_results, dtype=np.float64)
        n, bb = len(ev), self.engine.big_blind
        return {
            "ev_total": float(ev.sum()) if n else 0.0,
            "ev_avg": float(ev.mean()) if n else 0.0,
            "ev_mbb": float(ev.mean() * 1000 / bb) if n else 0.0,
            "ev_se_mbb": float(ev.std(ddof=1) / np.sqrt(n) * 1000 / bb) if n > 1 else 0.0,
            "allin_hands": int((ev != r).sum()) if n else 0,
            "cumulative_ev": np.cumsum(ev).tolist()[-400:] if n else [],
            **self._adj_stats(r),
        }

    def _adj_stats(self, r):
        """Luck-adjusted session: all-in runouts and the bot's actual hand at showdowns replaced by their averages."""
        adj = np.asarray(self.adj_results, dtype=np.float64)
        n, bb = len(adj), self.engine.big_blind
        return {
            "adj_total": float(adj.sum()) if n else 0.0,
            "adj_mbb": float(adj.mean() * 1000 / bb) if n else 0.0,
            "adj_se_mbb": float(adj.std(ddof=1) / np.sqrt(n) * 1000 / bb) if n > 1 else 0.0,
            "luck": float(r.sum() - adj.sum()) if n else 0.0,
            "adjusted_hands": int((adj != r).sum()) if n else 0,
            "cumulative_adj": np.cumsum(adj).tolist()[-400:] if n else [],
        }

    def state(self, reveal_bot=False):
        e = self.engine
        me, opp = self.me, self.opp
        showdown = self.hand_over and e.folded < 0
        return {
            "hand_number": e.hands_played,
            "stage": "showdown" if showdown else ("hand over" if self.hand_over else Stage(e.stage).name.lower()),
            "hand_over": self.hand_over,
            "your_turn": (not self.hand_over) and e.current == me,
            "bot_to_act": self.bot_to_act,
            "you": {
                "seat": me,
                "position": self._pos(me),
                "cards": [card_json(c) for c in e.hands[me]],
                "stack": e.stacks[me],
                "bet": 0 if self.hand_over else e.stage_bets[me],
                "total_bet": e.bets[me],
                "hand_class": describe_hand(e.hands[me], e.visible_board),
                "folded": e.folded == me,
            },
            "bot": {
                "seat": opp,
                "name": self.opponent_name,
                "label": bot_label(self.opponent_name),
                "position": self._pos(opp),
                "cards": [card_json(c) for c in e.hands[opp]] if (showdown or reveal_bot) else [{"hidden": True}, {"hidden": True}],
                "stack": e.stacks[opp],
                "bet": 0 if self.hand_over else e.stage_bets[opp],
                "total_bet": e.bets[opp],
                "folded": e.folded == opp,
                "last": self.bot_last,
                "hand_class": describe_hand(e.hands[opp], e.board) if showdown else None,
            },
            "board": [card_json(c) for c in (e.board if showdown else e.visible_board)],
            "pot": e.pot,
            "to_call": max(0, e.stage_bets[opp] - e.stage_bets[me]),
            "dealer_seat": e.dealer,
            "actions": self.legal_actions(),
            "action_names": self.action_names,
            "action_labels": self.action_labels,
            "result": self.result,
            "log": self.log,
            "history": self.hand_records[-30:][::-1],
            "stats": self.stats(),
            "settings": self.settings,
            "warning": self.warning,
            "advisor": {"loaded": self.advisor is not None, "name": self.settings["advisor"], "error": self.advisor_error},
        }

    def advice(self, equity_samples=2000):
        """Advisor probabilities for the human's spot + Monte-Carlo equity of the human's hand."""
        e = self.engine
        out = {"probs": None, "equity": None}
        if self.hand_over:
            return out
        if self.advisor is not None and e.current == self.me:
            probs = self.advisor.probs(e.observation(self.me)[None])[0]
            out["probs"] = [{"action": a, "label": l, "p": float(p)} for a, l, p in zip(self.action_names, self.action_labels, probs)]
        key = (tuple(e.hands[self.me]), tuple(e.visible_board))
        if key not in self._equity_cache:
            win, tie = equity(e.hands[self.me], e.visible_board, self.rng, equity_samples)
            self._equity_cache[key] = {"win": win, "tie": tie, "samples": equity_samples}
        out["equity"] = self._equity_cache[key]
        return out
