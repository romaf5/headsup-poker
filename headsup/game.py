"""Game (action-tree) configuration shared by the engines, the networks and the players.

Actions are ``0 FOLD, 1 CHECK_CALL, 2 .. 2+K-1 RAISE_k[, 2+K ALL_IN]`` where the K raise sizes
are the ``bet_sizes`` of the game: ``"min"`` (call + one big blind, the original game) or a
fraction of the pot after calling (0.5 = half pot, 1 = pot, ...), always at least a min-raise
and at most the stack (a raise that reaches the stack is an all-in).  ``raise_cap`` consecutive
raises turn the last one into an all-in (keeps the tree finite).

Limit games (FHP / HULH of the DeepCFR paper, Appendix A) use ``limit`` = fixed raise increments
per betting round, ``raise_caps`` per round and no all-in action (``all_in=False``); a raise past
the cap is simply not available.  ``num_rounds`` = 2 for FHP (pre-flop + flop, showdown on 5 cards).

``mask_redundant`` hides raise actions that would only duplicate another action - reaching the
stack (= ALL_IN), equal to a smaller size after rounding, or the cap-th raise (= ALL_IN) - so
CFR does not split regret between identical branches.  The default game keeps them visible
(they are executed as the action they duplicate) for continuity with the runs trained that way.

The part of the configuration that fixes the action tree (``bet_sizes``, ``raise_cap``,
``mask_redundant``) is stored in every model's config; envs / players build engines from it.
"""

from dataclasses import asdict, dataclass, replace

MAX_ACTIONS = 8  # mirrored by headsup_cpp (MAX_ACTIONS): up to 5 raise sizes


def parse_bet_sizes(text):
    """``"min"`` | ``"0.5,1,2"`` | ``"min,1"`` -> tuple of "min" / floats (as given, in order)."""
    if isinstance(text, (list, tuple)):
        return tuple("min" if s == "min" else float(s) for s in text)
    sizes = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append("min" if part == "min" else float(part))
    if not sizes:
        raise ValueError("at least one bet size is needed")
    if len(sizes) + 3 > MAX_ACTIONS:
        raise ValueError(f"at most {MAX_ACTIONS - 3} bet sizes are supported")
    return tuple(sizes)


@dataclass(frozen=True)
class GameConfig:
    stack_size: int = 100
    small_blind: int = 1
    big_blind: int = 2
    raise_cap: int = 3
    bet_sizes: tuple = ("min",)
    mask_redundant: bool = False
    limit: tuple | None = None       # fixed raise increment per round (limit poker); overrides bet_sizes
    raise_caps: tuple | None = None  # per-round raise caps (default: raise_cap everywhere)
    num_rounds: int = 4              # betting rounds (2 = FHP: showdown after the flop)
    all_in: bool = True              # whether the ALL_IN action exists (limit games: no)

    def __post_init__(self):
        if self.limit is not None:
            object.__setattr__(self, "limit", tuple(int(x) for x in self.limit))
            object.__setattr__(self, "bet_sizes", ("limit",))
            if len(self.limit) != self.num_rounds:
                raise ValueError("limit needs one raise increment per round")
        else:
            object.__setattr__(self, "bet_sizes", parse_bet_sizes(self.bet_sizes))
        if self.raise_caps is not None:
            object.__setattr__(self, "raise_caps", tuple(int(x) for x in self.raise_caps))
            if len(self.raise_caps) != self.num_rounds:
                raise ValueError("raise_caps needs one entry per round")
        assert 0 < self.small_blind < self.big_blind < self.stack_size
        assert 1 <= self.num_rounds <= 4

    @property
    def num_raises(self):
        return len(self.bet_sizes)

    @property
    def num_actions(self):
        return 2 + self.num_raises + (1 if self.all_in else 0)

    @property
    def all_in_action(self):
        """Index of the all-in action, or None in games without one."""
        return 2 + self.num_raises if self.all_in else None

    def is_raise(self, action):
        return 2 <= action < 2 + self.num_raises

    def cap(self, round_index):
        return self.raise_caps[round_index] if self.raise_caps is not None else self.raise_cap

    def raise_amount(self, k, to_call, pot, stack, round_index=0):
        """Chips the acting player puts in for raise action index k (already capped by the stack)."""
        size = self.bet_sizes[k - 2]
        if size == "limit":
            return min(to_call + self.limit[round_index], stack)
        min_raise = to_call + self.big_blind
        if size == "min":
            amount = min_raise
        else:
            amount = max(min_raise, to_call + int(round(size * (pot + to_call))))
        return min(amount, stack)

    def legal_mask(self, to_call, pot, stack, consecutive_raises, with_twins=False, round_index=0):
        """bool per action: FOLD only when facing a bet; raises per ``mask_redundant`` (and, without
        an all-in action, never past the round's cap or the stack).  With ``with_twins`` also the
        action each redundant one duplicates (itself when legal)."""
        n = self.num_actions
        mask = [to_call > 0, True] + [True] * self.num_raises + ([True] if self.all_in else [])
        twins = list(range(n))
        if to_call <= 0:
            twins[0] = 1
        # no-limit: the cap-th raise in a row is executed as an all-in (historic tree, kept as is);
        # limit: at most cap raise actions per round, the next one is simply unavailable
        capped = (consecutive_raises + 1 >= self.cap(round_index)) if self.all_in else (consecutive_raises >= self.cap(round_index))
        collapse = self.all_in_action if self.all_in else 1  # a raise that cannot happen "is" an all-in / a call
        if self.mask_redundant or not self.all_in:
            seen = {}
            for k in range(2, 2 + self.num_raises):
                amount = self.raise_amount(k, to_call, pot, stack, round_index)
                if capped or (amount >= stack and self.all_in) or (stack <= to_call):
                    mask[k], twins[k] = False, collapse
                elif self.mask_redundant and amount in seen:
                    mask[k], twins[k] = False, twins[seen[amount]]
                seen.setdefault(amount, k)
        return (mask, twins) if with_twins else mask

    # -- (de)serialisation ------------------------------------------------------------
    def tree_dict(self):
        """The part that defines the action tree (stored in model configs).  Stacks and blinds are
        included when they differ from the no-limit defaults: for limit games the bet sizes are
        absolute chips, so blinds / stacks are part of the tree (a model trained on FHP must not be
        evaluated in a 100-chip-stack, 1/2-blind game)."""
        d = {"bet_sizes": list(self.bet_sizes), "raise_cap": self.raise_cap, "mask_redundant": self.mask_redundant}
        if self.limit is not None or self.raise_caps is not None or self.num_rounds != 4 or not self.all_in:
            d.update({"limit": list(self.limit) if self.limit else None, "raise_caps": list(self.raise_caps) if self.raise_caps else None,
                      "num_rounds": self.num_rounds, "all_in": self.all_in})
        if (self.stack_size, self.small_blind, self.big_blind) != (100, 1, 2):
            d.update({"stack_size": self.stack_size, "small_blind": self.small_blind, "big_blind": self.big_blind})
        return d

    def to_dict(self):
        d = asdict(self)
        d["bet_sizes"] = list(self.bet_sizes)
        for key in ("limit", "raise_caps"):
            if d[key] is not None:
                d[key] = list(d[key])
        return d

    @staticmethod
    def from_dict(d, **overrides):
        d = {**(d or {}), **overrides}
        return GameConfig(**{k: v for k, v in d.items() if v is not None})

    def with_(self, **kw):
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


DEFAULT_GAME = GameConfig()
# DeepCFR paper, Appendix A: blinds 50/100, raises of $100 in rounds 1-2 ($200 in 3-4), at most 3 raises
# per round in rounds 1-2 (4 in 3-4), no all-in (stacks are never a constraint); FHP = the first two rounds
FHP = GameConfig(stack_size=100_000, small_blind=50, big_blind=100, limit=(100, 100), raise_caps=(3, 3), num_rounds=2, all_in=False)
HULH = GameConfig(stack_size=100_000, small_blind=50, big_blind=100, limit=(100, 100, 200, 200), raise_caps=(3, 3, 4, 4), num_rounds=4, all_in=False)


def action_label(game, action, verbose=False):
    """Short generic name of an action index in this game (``fold``, ``call``, ``raise_1p``, ...)."""
    if action == 0:
        return "fold"
    if action == 1:
        return "call"
    if action == game.all_in_action:
        return "allin"
    size = game.bet_sizes[action - 2]
    if size in ("min", "limit"):
        return "raise" if len(game.bet_sizes) == 1 else "raise_min"
    return f"raise_{size:g}p"
