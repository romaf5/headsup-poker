"""Loader for the optional C++ extension (``headsup_cpp``).

Build it with ``python setup.py build_ext --inplace``.  On import success the treys hand
ranking tables are installed into the extension so its showdown results match
:func:`headsup.cards.hand_strength` exactly.
"""

import numpy as np

try:
    import headsup_cpp as _cpp
except ImportError:  # pragma: no cover - extension not built
    _cpp = None

_installed = False


def _install_tables():
    global _installed
    if _installed or _cpp is None:
        return
    from treys.lookup import LookupTable

    table = LookupTable()
    fk = np.fromiter(table.flush_lookup.keys(), dtype=np.uint32, count=len(table.flush_lookup))
    fv = np.fromiter(table.flush_lookup.values(), dtype=np.uint16, count=len(table.flush_lookup))
    uk = np.fromiter(table.unsuited_lookup.keys(), dtype=np.uint32, count=len(table.unsuited_lookup))
    uv = np.fromiter(table.unsuited_lookup.values(), dtype=np.uint16, count=len(table.unsuited_lookup))
    _cpp.set_tables(fk, fv, uk, uv)
    _installed = True


def available() -> bool:
    return _cpp is not None


def module():
    """Return the extension module (tables installed) or raise ImportError."""
    if _cpp is None:
        raise ImportError(
            "headsup_cpp extension is not built; run `python setup.py build_ext --inplace`"
        )
    _install_tables()
    return _cpp


def engine_config(stack_size=100, small_blind=1, big_blind=2, raise_cap=3):
    cfg = module().EngineConfig()
    cfg.stack_size = stack_size
    cfg.small_blind = small_blind
    cfg.big_blind = big_blind
    cfg.raise_cap = raise_cap
    return cfg


def make_model(weights: dict):
    """Wrap numpy weights (``BaseModel.numpy_weights()``) into a C++ model."""
    return module().Model({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()})
