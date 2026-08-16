"""Game (action-tree) configuration shared by the engines, the networks and the players.

Actions are ``0 FOLD, 1 CHECK_CALL, 2 .. 2+K-1 RAISE_k, 2+K ALL_IN`` where the K raise sizes
are the ``bet_sizes`` of the game: ``"min"`` (call + one big blind, the original game) or a
fraction of the pot after calling (0.5 = half pot, 1 = pot, ...), always at least a min-raise
and at most the stack (a raise that reaches the stack is an all-in).  ``raise_cap`` consecutive
raises turn the last one into an all-in (keeps the tree finite).

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

    def __post_init__(self):
        object.__setattr__(self, "bet_sizes", parse_bet_sizes(self.bet_sizes))
        assert 0 < self.small_blind < self.big_blind < self.stack_size

    @property
    def num_actions(self):
        return len(self.bet_sizes) + 3

    @property
    def all_in(self):
        return self.num_actions - 1

    def is_raise(self, action):
        return 2 <= action < self.all_in

    def raise_amount(self, k, to_call, pot, stack):
        """Chips the acting player puts in for raise action index k (already capped by the stack)."""
        size = self.bet_sizes[k - 2]
        min_raise = to_call + self.big_blind
        if size == "min":
            amount = min_raise
        else:
            amount = max(min_raise, to_call + int(round(size * (pot + to_call))))
        return min(amount, stack)

    def legal_mask(self, to_call, pot, stack, consecutive_raises, with_twins=False):
        """bool per action: FOLD only when facing a bet; raises per ``mask_redundant``.  With
        ``with_twins`` also the action each redundant one duplicates (itself when legal)."""
        mask = [to_call > 0, True] + [True] * len(self.bet_sizes) + [True]
        twins = list(range(self.num_actions))
        if to_call <= 0:
            twins[0] = 1
        if self.mask_redundant:
            seen = {}
            for k in range(2, self.all_in):
                amount = self.raise_amount(k, to_call, pot, stack)
                if amount >= stack or consecutive_raises + 1 >= self.raise_cap:
                    mask[k], twins[k] = False, self.all_in
                elif amount in seen:
                    mask[k], twins[k] = False, twins[seen[amount]]
                seen.setdefault(amount, k)
        return (mask, twins) if with_twins else mask

    # -- (de)serialisation ------------------------------------------------------------
    def tree_dict(self):
        """The part that defines the action tree (stored in model configs)."""
        return {"bet_sizes": list(self.bet_sizes), "raise_cap": self.raise_cap, "mask_redundant": self.mask_redundant}

    def to_dict(self):
        d = asdict(self)
        d["bet_sizes"] = list(self.bet_sizes)
        return d

    @staticmethod
    def from_dict(d, **overrides):
        d = {**(d or {}), **overrides}
        return GameConfig(**{k: v for k, v in d.items() if v is not None})

    def with_(self, **kw):
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


DEFAULT_GAME = GameConfig()


def action_label(game, action, verbose=False):
    """Short generic name of an action index in this game (``fold``, ``call``, ``raise_1p``, ...)."""
    if action == 0:
        return "fold"
    if action == 1:
        return "call"
    if action == game.all_in:
        return "allin"
    size = game.bet_sizes[action - 2]
    if size == "min":
        return "raise" if len(game.bet_sizes) == 1 else "raise_min"
    return f"raise_{size:g}p"
