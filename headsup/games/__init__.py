"""Games: ``make_game("leduc")``, ``make_game("kuhn")``; hold'em variants live in headsup.games.holdem."""

from headsup.games.base import CHANCE, TERMINAL, Game, State, UniformPolicy  # noqa: F401
from headsup.games.leduc import Kuhn, Leduc

_REGISTRY = {"leduc": Leduc, "kuhn": Kuhn}


def make_game(name, **kwargs):
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if name in ("holdem", "nlhe", "fhp", "hulh"):  # GameConfig presets for the hold'em engine
        from headsup.games.holdem import make_holdem

        return make_holdem(name, **kwargs)
    raise ValueError(f"unknown game {name!r}; known: {sorted(_REGISTRY)} + holdem/fhp/hulh")
