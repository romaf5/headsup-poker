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


def make_small_game(name, **kwargs):
    """A game the tree-enumerating solvers can hold in memory (their CLIs' ``--game``)."""
    if name not in _REGISTRY:
        raise ValueError(f"{name!r} is not a small game ({' / '.join(sorted(_REGISTRY))}): these solvers enumerate the game tree; "
                         "hold'em is trained with headsup.deepcfr.train and solved exactly (FHP) with headsup.algos.fhp_cfr")
    return _REGISTRY[name](**kwargs)
