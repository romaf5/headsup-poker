"""Single-agent environments: the agent plays hands against a fixed (batched) opponent.

``PokerVecEnv`` runs ``num_envs`` tables in lock-step and asks the opponent for all its
decisions in one batched call, which is what makes evaluation and rl_games training fast.
``SingleAgentEnv`` is the one-table convenience wrapper with the classic gym-style API
(``reset() -> obs``, ``step(a) -> (obs, reward, done, info)``).

Seats alternate between hands by default (``seat_mode="alternate"``) so both positions are
sampled equally; ``"random"`` reproduces the original environment's behaviour.

Note on pre-flop folds: if the opponent is the small blind and open-folds, ``reset()``
already returns a finished hand.  The observation is still valid; the very next ``step``
(with any action) returns the +small-blind reward and ``done=True``.  This keeps every
hand's result in the reward stream instead of silently re-dealing.
"""

import numpy as np

from headsup.engine import OBS_DIM, HeadsUpPoker
from headsup.enums import NUM_ACTIONS

try:  # spaces are only needed for RL libraries
    from gymnasium import spaces as _spaces
except ImportError:  # pragma: no cover
    _spaces = None


def _call_player(player, obs, ids):
    """Call a batched player, passing table ids to players that track per-table state."""
    if getattr(player, "wants_ids", False):
        return player(obs, ids=ids)
    return player(obs)


def _make_spaces():
    if _spaces is None:
        return None, None
    return (
        _spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32),
        _spaces.Discrete(NUM_ACTIONS),
    )


class PokerVecEnv:
    def __init__(
        self,
        num_envs,
        opponent,
        seat_mode="alternate",
        seed=None,
        **engine_kwargs,
    ):
        assert seat_mode in ("alternate", "random")
        self.num_envs = num_envs
        self.opponent = opponent
        self.seat_mode = seat_mode
        self.rng = np.random.default_rng(seed)
        self.engines = [
            HeadsUpPoker(rng=np.random.default_rng(self.rng.integers(2**63)), **engine_kwargs)
            for _ in range(num_envs)
        ]
        # balanced initial seats (flipped on the first reset, so env 0 starts as dealer)
        self.agent_seat = (np.arange(num_envs) % 2) ^ 1
        self.observation_space, self.action_space = _make_spaces()
        self.hands_completed = 0

    # ---------------------------------------------------------------- helpers
    def _reset_engine(self, i):
        if self.seat_mode == "alternate":
            self.agent_seat[i] ^= 1
        else:
            self.agent_seat[i] = self.rng.integers(2)
        self.engines[i].reset()

    def _advance_opponent(self):
        """Let the opponent act (batched) until every table waits for the agent or is done."""
        while True:
            idx = [
                i
                for i, e in enumerate(self.engines)
                if not e.done and e.current != self.agent_seat[i]
            ]
            if not idx:
                return
            obs = np.stack([self.engines[i].observation() for i in idx])
            actions = _call_player(self.opponent, obs, np.asarray(idx))
            for i, a in zip(idx, actions):
                self.engines[i].step(int(a))

    def _agent_obs(self):
        return np.stack(
            [e.observation(int(s)) for e, s in zip(self.engines, self.agent_seat)]
        )

    # ---------------------------------------------------------------- API
    def reset(self):
        for i in range(self.num_envs):
            self._reset_engine(i)
        self._advance_opponent()
        return self._agent_obs()

    def step(self, actions, auto_reset=True):
        actions = np.asarray(actions).reshape(-1)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for i, (e, a) in enumerate(zip(self.engines, actions)):
            if e.done:  # opponent open-folded during reset: any action collects the pot
                dones[i] = True
                continue
            e.step(int(a))
        self._advance_opponent()
        for i, e in enumerate(self.engines):
            if e.done:
                dones[i] = True
                rewards[i] = e.rewards[self.agent_seat[i]]
        self.hands_completed += int(dones.sum())
        infos = [{} for _ in range(self.num_envs)]
        if auto_reset:
            for i in np.flatnonzero(dones):
                self._reset_engine(i)
            self._advance_opponent()
        return self._agent_obs(), rewards, dones, infos

    # rl_games IVecEnv interface -------------------------------------------------
    def get_number_of_agents(self):
        return 1

    def has_action_masks(self):
        return False

    def get_env_info(self):
        return {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "state_space": None,
            "use_global_observations": False,
            "agents": 1,
            "value_size": 1,
        }

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        for e in self.engines:
            e.rng = np.random.default_rng(self.rng.integers(2**63))

    def set_train_info(self, *args, **kwargs):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass

    def close(self):
        pass


class SingleAgentEnv:
    """One table, gym-style API, agent vs. a fixed opponent."""

    def __init__(self, opponent, seat_mode="alternate", seed=None, **engine_kwargs):
        self.vec = PokerVecEnv(1, opponent, seat_mode=seat_mode, seed=seed, **engine_kwargs)
        self.observation_space = self.vec.observation_space
        self.action_space = self.vec.action_space

    @property
    def engine(self) -> HeadsUpPoker:
        return self.vec.engines[0]

    @property
    def agent_seat(self) -> int:
        return int(self.vec.agent_seat[0])

    @property
    def opponent_seat(self) -> int:
        return 1 - self.agent_seat

    @property
    def opponent(self):
        return self.vec.opponent

    def reset(self):
        return self.vec.reset()[0]

    def step(self, action):
        obs, rewards, dones, infos = self.vec.step([action], auto_reset=False)
        return obs[0], float(rewards[0]), bool(dones[0]), infos[0]

    def render(self):
        print(self.engine.describe(self.agent_seat))

    def close(self):
        pass


def play_hands(vec_env, agent, num_hands, progress=False):
    """Run ``num_hands`` complete hands of ``agent`` in ``vec_env``; returns per-hand rewards."""
    rewards = []
    obs = vec_env.reset()
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=num_hands, leave=False)
    ids = np.arange(vec_env.num_envs)
    while len(rewards) < num_hands:
        obs, r, d, _ = vec_env.step(_call_player(agent, obs, ids))
        got = r[d]
        rewards.extend(got.tolist())
        if bar is not None:
            bar.update(len(got))
    if bar is not None:
        bar.close()
    return np.asarray(rewards[:num_hands], dtype=np.float32)


class NativeVecEnv:
    """envpool-style vectorised env backed by the C++ extension (same interface as PokerVecEnv).

    The opponent runs inside C++ too: either a simple bot ("random", "call", "allin",
    "raise") or a :class:`headsup.model.BaseModel` given by its numpy weights.
    """

    def __init__(self, num_envs, opponent="call", seat_mode="alternate", seed=None, deterministic=False, **engine_kwargs):
        from headsup import native

        self._cpp = native.module()
        seed = int(np.random.default_rng(seed).integers(2**63)) if seed is None else int(seed)
        self.num_envs = num_envs
        self.env = self._cpp.VecEnv(num_envs, seed, native.engine_config(**engine_kwargs), seat_mode == "alternate")
        self.set_opponent(opponent, deterministic)
        self.observation_space, self.action_space = _make_spaces()

    def set_opponent(self, opponent, deterministic=False):
        from headsup import native

        if isinstance(opponent, str):
            self.env.set_opponent_simple(opponent)
        elif isinstance(opponent, dict):  # numpy weights
            self.env.set_opponent_model(native.make_model(opponent), deterministic)
        else:  # torch model
            self.env.set_opponent_model(native.make_model(opponent.numpy_weights()), deterministic)
        self.opponent = opponent

    @property
    def agent_seat(self):
        return np.asarray(self.env.agent_seat)

    @property
    def hands_completed(self):
        return self.env.hands_completed

    @property
    def opponent_last_probs(self):
        return self.env.last_probs

    def engine_state(self, i=0):
        return self.env.engine_state(i)

    def reset(self):
        return self.env.reset()

    def step(self, actions, auto_reset=True):
        obs, rewards, dones = self.env.step(np.asarray(actions, dtype=np.int64).reshape(-1), auto_reset)
        return obs, rewards, dones, [{} for _ in range(self.num_envs)]

    # rl_games IVecEnv interface
    def get_number_of_agents(self):
        return 1

    def has_action_masks(self):
        return False

    def get_env_info(self):
        return {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "state_space": None,
            "use_global_observations": False,
            "agents": 1,
            "value_size": 1,
        }

    def seed(self, seed):
        pass

    def set_train_info(self, *args, **kwargs):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass

    def close(self):
        pass


def make_vec_env(num_envs, opponent, seat_mode="alternate", seed=None, backend="auto", deterministic=False, **engine_kwargs):
    """Create a vectorised env.

    ``opponent`` may be a player spec string understood by :func:`headsup.players.make_player`
    ("random", "call", "allin", "raise", "cfr[:path]", "onnx[:path]"), a batched player
    callable, a :class:`BaseModel` or its numpy weights.  ``backend`` "cpp" uses the
    extension when the opponent can run natively (simple bots and BaseModel weights).
    """
    from headsup import native

    native_ok = native.available() and backend != "python"
    native_opponent = isinstance(opponent, dict) or (isinstance(opponent, str) and opponent in ("random", "call", "allin", "raise"))
    if not native_opponent and hasattr(opponent, "numpy_weights"):
        native_opponent = True
    if native_ok and native_opponent:
        return NativeVecEnv(num_envs, opponent, seat_mode=seat_mode, seed=seed, deterministic=deterministic, **engine_kwargs)
    if backend == "cpp":
        raise ValueError("cpp backend requested but the opponent cannot run natively (or extension missing)")
    if isinstance(opponent, str):
        from headsup.players import make_player

        opponent = make_player(opponent, deterministic=deterministic, seed=seed)
    elif hasattr(opponent, "numpy_weights"):
        from headsup.players import TorchPolicyPlayer

        opponent = TorchPolicyPlayer(opponent, deterministic=deterministic, seed=seed)
    return PokerVecEnv(num_envs, opponent, seat_mode=seat_mode, seed=seed, **engine_kwargs)
