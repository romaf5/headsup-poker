"""Hold'em variants as game configurations: the no-limit abstraction (``nlhe``, the default
game), Flop Hold'em Poker (``fhp``) and heads-up limit hold'em (``hulh``) with the DeepCFR
paper's rules (Appendix A).  These are :class:`headsup.game.GameConfig` presets for the fast
engine (headsup.engine / headsup_cpp); the deep pipeline for them is headsup.deepcfr.train
(``--game fhp``), the exploitability estimator headsup.algos.holdem_br.
"""

from headsup.game import DEFAULT_GAME, FHP, HULH, GameConfig

PRESETS = {"nlhe": DEFAULT_GAME, "holdem": DEFAULT_GAME, "fhp": FHP, "hulh": HULH}


def make_holdem(name, **overrides):
    """GameConfig preset by name (``nlhe`` / ``fhp`` / ``hulh``) with optional field overrides."""
    if name not in PRESETS:
        raise ValueError(f"unknown hold'em game {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name].with_(**overrides) if overrides else PRESETS[name]


def mbb_per_hand(chips, game: GameConfig):
    """Chips per hand -> milli big blinds per game (the papers' unit)."""
    return 1000.0 * chips / game.big_blind
