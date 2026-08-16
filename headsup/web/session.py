"""Game session behind the web UI: one human seat vs a bot, with history, advisor and stats.

The session drives :class:`headsup.engine.HeadsUpPoker` directly (rather than through
``SingleAgentEnv``) so every bot action can be recorded with its amount for the hand log.
"""

import os
import threading
import time

import numpy as np

from headsup.cards import card_to_str, describe_hand, hand_strength
from headsup.engine import HeadsUpPoker
from headsup.enums import Action, Stage
from headsup.paths import DEFAULT_POLICY_PATH
from headsup.players import make_player

ACTION_NAMES = {Action.FOLD: "fold", Action.CHECK_CALL: "call", Action.RAISE: "raise", Action.ALL_IN: "all-in"}
BOT_LABELS = {
    "cfr": "DeepCFR", "sdcfr": "SD-CFR", "onnx": "PPO exploiter", "random": "Random bot", "call": "Calling station",
    "allin": "Maniac", "raise": "Always-raise bot",
}


def bot_label(spec):
    kind, _, arg = str(spec).partition(":")
    if kind in BOT_LABELS and not arg:
        return BOT_LABELS[kind]
    name = os.path.basename(arg or spec)
    return f"{BOT_LABELS.get(kind, kind)} · {name}" if kind in BOT_LABELS else name
SUIT_GLYPH = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RANK_LABEL = {"T": "10"}


ACTION_ALIASES = {
    "fold": Action.FOLD, "check": Action.CHECK_CALL, "call": Action.CHECK_CALL, "check_call": Action.CHECK_CALL,
    "raise": Action.RAISE, "bet": Action.RAISE, "allin": Action.ALL_IN, "all_in": Action.ALL_IN, "all-in": Action.ALL_IN,
}


def parse_action(value):
    if isinstance(value, (int, np.integer)) or str(value).lstrip("-").isdigit():
        return Action(int(value))
    key = str(value).strip().lower()
    if key not in ACTION_ALIASES:
        raise ValueError(f"unknown action {value!r}")
    return ACTION_ALIASES[key]


def player_spec(value):
    """Accept player specs ('cfr', 'onnx:path', 'sdcfr:path', 'call', ...) as well as bare model paths."""
    value = str(value).strip()
    if ":" in value or value in ("cfr", "onnx", "random", "call", "allin", "raise"):
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
        stack_size=100,
        small_blind=1,
        big_blind=2,
        raise_cap=3,
        seed=None,
        deterministic=False,
        device=None,
    ):
        self.lock = threading.RLock()
        if advisor and os.path.abspath(str(advisor)) == os.path.abspath(DEFAULT_POLICY_PATH):
            advisor = "cfr"
        self.settings = dict(
            opponent=opponent,
            advisor=advisor,
            stack_size=int(stack_size),
            small_blind=int(small_blind),
            big_blind=int(big_blind),
            raise_cap=int(raise_cap),
            seed=seed,
            deterministic=bool(deterministic),
        )
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.engine = HeadsUpPoker(
            stack_size=int(stack_size),
            small_blind=int(small_blind),
            big_blind=int(big_blind),
            raise_cap=int(raise_cap),
            rng=np.random.default_rng(self.rng.integers(2**63)),
        )
        self.opponent = make_player(player_spec(opponent), device=device, deterministic=deterministic, seed=int(self.rng.integers(2**31)))
        self.opponent_name = opponent
        self.advisor = None
        self.advisor_error = None
        if advisor:
            try:
                self.advisor = make_player(player_spec(advisor), device=device, seed=int(self.rng.integers(2**31)))
            except Exception as exc:  # advisor is optional
                self.advisor_error = f"{type(exc).__name__}: {exc}"
        self.me = 1  # flipped in new_hand -> human is dealer first
        self.results = []  # per-hand rewards for the human
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
        action = Action(action)
        if action == Action.FOLD:
            return "folds", 0
        if action == Action.CHECK_CALL:
            amt = min(to_call, stack)
            return ("checks", 0) if amt == 0 else (f"calls {amt}", amt)
        if action == Action.RAISE:
            verb = "bets" if to_call == 0 else "raises"
            if before["consecutive_raises"] + 1 >= e.raise_cap or to_call + e.big_blind >= stack:
                return f"{verb} all-in {stack}", stack
            amt = to_call + e.big_blind
            return (f"bets {amt}", amt) if to_call == 0 else (f"raises to {before['stage_bets'][seat] + amt}", amt)
        return f"all-in {stack}", stack

    def _snapshot(self):
        e = self.engine
        return dict(stage_bets=list(e.stage_bets), stacks=list(e.stacks), consecutive_raises=e.consecutive_raises)

    def _record(self, seat, action, before):
        text, amount = self._describe_action(seat, action, before)
        self.log.append(
            {
                "seat": "you" if seat == self.me else "bot",
                "stage": Stage(self.engine.stage).name.lower() if not self.engine.done else "end",
                "action": ACTION_NAMES[Action(action)],
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
            "action": ACTION_NAMES[Action(action)],
            "text": self._describe_action(self.opp, action, self._snapshot())[0],
            "probs": [float(p) for p in np.asarray(probs).reshape(-1)] if probs is not None else None,
            "hand": e.hands_played,
            "n": len(self.log),
        }
        self._step(self.opp, action)
        return self.state()

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
            }
        )
        self.hand_records = self.hand_records[-200:]

    # ------------------------------------------------------------------ actions
    def new_hand(self):
        self.me ^= 1
        self.engine.reset()
        self.hand_over = False
        self.result = None
        self.log = []
        self.bot_last = None
        self._equity_cache = {}
        return self.state()

    def act(self, action):
        if self.hand_over:
            raise ValueError("hand is over — start the next hand")
        action = parse_action(action)
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
        out = [{"action": "fold", "label": "Fold", "key": "F"}] if to_call > 0 else []
        out.append({"action": "call", "label": "Check" if to_call == 0 else f"Call {min(to_call, stack)}", "key": "C"})
        raise_amt = to_call + e.big_blind
        verb = "Bet" if to_call == 0 else "Raise"
        if e.consecutive_raises + 1 >= e.raise_cap or raise_amt >= stack:
            out.append({"action": "raise", "label": f"{verb} all-in {stack}", "key": "R", "allin": True})
        elif to_call == 0:
            out.append({"action": "raise", "label": f"Bet {raise_amt}", "key": "R"})
        else:
            out.append({"action": "raise", "label": f"Raise to {e.stage_bets[me] + raise_amt}", "key": "R"})
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
            "result": self.result,
            "log": self.log,
            "history": self.hand_records[-30:][::-1],
            "stats": self.stats(),
            "settings": self.settings,
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
            labels = ["Fold", "Check/Call", "Raise", "All-in"]
            out["probs"] = [{"action": a, "label": l, "p": float(p)} for a, l, p in zip(["fold", "call", "raise", "allin"], labels, probs)]
        key = (tuple(e.hands[self.me]), tuple(e.visible_board))
        if key not in self._equity_cache:
            win, tie = equity(e.hands[self.me], e.visible_board, self.rng, equity_samples)
            self._equity_cache[key] = {"win": win, "tie": tie, "samples": equity_samples}
        out["equity"] = self._equity_cache[key]
        return out
