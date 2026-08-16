"""rl_games integration: registers the ``headsup_poker`` environment (import this module).

Importing this module registers
* an env config ``headsup_poker`` (``env_configurations``) whose vecenv type ``HEADSUP``
  is our own vectorised environment (all tables in one process, opponent inference batched
  or fully native in C++), so no Ray workers are needed;
* a gym-style single env :class:`HeadsUpPokerRLGames` for tools that want one table.

``env_config`` keys (from the yaml): ``opponent`` (player spec, default ``cfr``),
``deterministic``, ``seed``, ``seat_mode`` and engine parameters.
"""


from headsup.env import SingleAgentEnv, make_vec_env
from headsup.players import make_player

ENV_NAME = "headsup_poker"
VECENV_TYPE = "HEADSUP"


def _split_config(kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("name", None)
    opponent = kwargs.pop("opponent", "cfr")
    deterministic = bool(kwargs.pop("deterministic", False))
    seed = kwargs.pop("seed", None)
    seat_mode = kwargs.pop("seat_mode", "alternate")
    device = kwargs.pop("device", None)
    return opponent, deterministic, seed, seat_mode, device, kwargs


class HeadsUpPokerRLGames(SingleAgentEnv):
    """One table with the classic gym API (obs float32[31], Discrete(4))."""

    def __init__(self, **kwargs):
        opponent, deterministic, seed, seat_mode, device, engine_kwargs = _split_config(kwargs)
        player = make_player(opponent, device=device or "cpu", deterministic=deterministic, seed=seed)
        super().__init__(player, seat_mode=seat_mode, seed=seed, **engine_kwargs)


def create_vec_env(config_name, num_actors, **kwargs):
    opponent, deterministic, seed, seat_mode, device, engine_kwargs = _split_config(kwargs)
    return make_vec_env(num_actors, opponent, seat_mode=seat_mode, seed=seed, deterministic=deterministic, **engine_kwargs)


def enable_mps_compat():
    """rl_games keeps float64 running statistics, which MPS cannot hold; use float32 instead."""
    import torch
    from rl_games.algos_torch import running_mean_std as rms

    if getattr(rms.RunningMeanStd, "_headsup_mps_patched", False):
        return
    original_init = rms.RunningMeanStd.__init__

    def patched_init(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        original_init(self, insize, epsilon, per_channel, norm_only)
        for name in ("running_mean", "running_var", "count"):
            buf = getattr(self, name)
            self._buffers[name] = buf.to(torch.float32)

    rms.RunningMeanStd.__init__ = patched_init
    rms.RunningMeanStd._headsup_mps_patched = True


def register():
    from rl_games.common import env_configurations, vecenv

    vecenv.register(VECENV_TYPE, create_vec_env)
    env_configurations.register(
        ENV_NAME,
        {"vecenv_type": VECENV_TYPE, "env_creator": lambda **kwargs: HeadsUpPokerRLGames(**kwargs)},
    )


try:
    register()
except ImportError:  # rl_games not installed: the classes above still work stand-alone
    pass


if __name__ == "__main__":
    env = HeadsUpPokerRLGames(opponent="call")
    obs = env.reset()
    total = 0.0
    for _ in range(20):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            total += reward
            print("hand over, reward", reward)
            obs = env.reset()
    print("total", total)
