"""Two-seat vectorised environment: both seats of every table are driven from Python, in lock-step.

The one-seat envs (:mod:`headsup.env`) play the opponent inside the env, which the C++ one can only do for a
built-in bot or one :class:`headsup.model.BaseModel`.  Self-play against several frozen networks needs the other
side of the table too, so here every ``step`` takes one action per table for the seat to act:

    obs, seat = env.reset()                          # float32[N, 80], int64[N]: the observation of the seat to act
    obs, seat, rewards, dones = env.step(actions)    # rewards float32[N, 2] (seat 0, seat 1), dones bool[N]

When an action ends a hand, ``rewards[i]`` holds both seats' chips, ``dones[i]`` is set, the table is dealt again
and ``obs[i]`` / ``seat[i]`` already belong to the new hand (seat 0 acts first).  ``actions[i] < 0`` lets table i
wait: nothing changes and the same observation comes back.  Nobody acts during a deal, so - unlike in the one-seat
envs - an observation is never terminal.

``decks`` (int[N, 9]: seat 0's hand, seat 1's hand, the board) fixes the cards of each table's next deal (tests);
without it the env deals from its own generator.  ``allin_ev``: hands that end all-in before the last round are
rewarded with their expectation over the runouts (``ev_samples`` sampled ones where there are more), as in
:class:`headsup.env.PokerVecEnv`; the sampling seed of a hand is a hash of ``ev_seed``, the table and the hand's
number, the same in both implementations.

:class:`TwoSeatVecEnv` runs Python engines (the reference; seeded like ``PokerVecEnv``, so the same seed deals the
same cards), :class:`NativeTwoSeatVecEnv` the C++ ``SelfPlayVecEnv`` (tests compare the two step by step).
"""

import numpy as np

from headsup.cards import NUM_CARDS
from headsup.engine import OBS_DIM, HeadsUpPoker
from headsup.env import resolve_game

_M64 = (1 << 64) - 1


def ev_hand_seed(ev_seed, table, hand):
    """Seed of the runout sampling of ``table``'s ``hand``-th hand (splitmix64 of the three; mirrored in C++)."""
    x = (int(ev_seed) + 0x9E3779B97F4A7C15 * ((int(table) << 32) + int(hand) + 1)) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def _check_decks(decks, n):
    decks = np.asarray(decks, dtype=np.int64)
    if decks.shape != (n, 9):
        raise ValueError(f"decks must have the shape ({n}, 9), not {decks.shape}")
    return decks


def _check_deck(deck):
    if len(set(int(c) for c in deck)) != 9 or min(deck) < 0 or max(deck) >= NUM_CARDS:
        raise ValueError("deck: 9 distinct cards in 0..51")


class TwoSeatVecEnv:
    """The Python reference implementation (slow: one engine object per table)."""

    def __init__(self, num_envs, seed=None, game=None, **engine_kwargs):
        self.num_envs = num_envs
        self.game = resolve_game(game, **engine_kwargs)
        self.rng = np.random.default_rng(seed)
        self.engines = [HeadsUpPoker(rng=np.random.default_rng(self.rng.integers(2**63)), game=self.game) for _ in range(num_envs)]
        self.hands_dealt = np.zeros(num_envs, dtype=np.int64)
        self.hands_completed = 0
        self.allin_ev, self.ev_samples = False, 1000
        self.ev_seed = int(self.rng.integers(2**63)) if seed is None else int(seed)

    def _deal(self, i, decks):
        if decks is not None:
            _check_deck(decks[i])
        self.engines[i].reset(None if decks is None else decks[i])
        self.hands_dealt[i] += 1

    def _output(self):
        obs = np.stack([e.observation() for e in self.engines]) if self.engines else np.zeros((0, OBS_DIM), np.float32)
        return obs, np.array([e.current for e in self.engines], dtype=np.int64)

    def reset(self, decks=None):
        decks = None if decks is None else _check_decks(decks, self.num_envs)
        for i in range(self.num_envs):
            self._deal(i, decks)
        return self._output()

    def step(self, actions, decks=None):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != self.num_envs:
            raise ValueError("actions has wrong length")
        decks = None if decks is None else _check_decks(decks, self.num_envs)
        for i, a in enumerate(actions):  # before anything is stepped
            if a >= self.game.num_actions:
                raise ValueError(f"Invalid action {a} at table {i}")
        rewards = np.zeros((self.num_envs, 2), dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for i, (e, a) in enumerate(zip(self.engines, actions)):
            if a < 0:  # the table waits
                continue
            e.step(int(a))
            if e.done:
                dones[i] = True
                if self.allin_ev:
                    rewards[i] = e.allin_ev(self.ev_samples, ev_hand_seed(self.ev_seed, i, self.hands_dealt[i]))
                else:
                    rewards[i] = e.rewards
                self.hands_completed += 1
                self._deal(i, decks)
        return (*self._output(), rewards, dones)


class NativeTwoSeatVecEnv:
    """The C++ ``SelfPlayVecEnv`` behind the same interface."""

    def __init__(self, num_envs, seed=None, game=None, **engine_kwargs):
        from headsup import native

        seed = int(np.random.default_rng(seed).integers(2**63)) if seed is None else int(seed)
        self.num_envs = num_envs
        self.game = resolve_game(game, **engine_kwargs)
        self.env = native.module().SelfPlayVecEnv(num_envs, seed, native.engine_config(game=self.game))

    @property
    def hands_completed(self):
        return self.env.hands_completed

    @property
    def allin_ev(self):
        return self.env.allin_ev

    @allin_ev.setter
    def allin_ev(self, on):
        self.env.allin_ev = bool(on)

    @property
    def ev_samples(self):
        return self.env.ev_samples

    @ev_samples.setter
    def ev_samples(self, n):
        self.env.ev_samples = int(n)

    @property
    def ev_seed(self):
        return self.env.ev_seed

    @ev_seed.setter
    def ev_seed(self, seed):
        self.env.ev_seed = int(seed)

    def reset(self, decks=None):
        return self.env.reset(None if decks is None else _check_decks(decks, self.num_envs))

    def step(self, actions, decks=None):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        return self.env.step(actions, None if decks is None else _check_decks(decks, self.num_envs))


def make_two_seat_env(num_envs, seed=None, backend="auto", game=None, **engine_kwargs):
    """The C++ env when the extension is built (``backend`` "auto" / "cpp"), else the Python one ("python")."""
    from headsup import native

    if backend == "cpp" or (backend == "auto" and native.available()):
        return NativeTwoSeatVecEnv(num_envs, seed=seed, game=game, **engine_kwargs)
    if backend not in ("auto", "python"):
        raise ValueError(f"unknown backend {backend!r}")
    return TwoSeatVecEnv(num_envs, seed=seed, game=game, **engine_kwargs)
