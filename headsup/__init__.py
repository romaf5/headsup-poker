"""Heads-up no-limit-style Texas Hold'em environment, DeepCFR trainer and tools.

Package layout
--------------
- ``headsup.engine``     – the two-player game engine (rules, chips, showdown).
- ``headsup.env``        – single-agent / vectorised envs that play against a fixed opponent.
- ``headsup.players``    – batched opponents (random, call, all-in, torch policy, onnx policy).
- ``headsup.model``      – the DeepCFR advantage/policy network (PyTorch).
- ``headsup.numpy_model``– a numpy mirror of the network used by CPU traversal workers.
- ``headsup.deepcfr``    – reservoir memories, traversals, training and evaluation.
"""

from headsup.enums import Action, Stage  # noqa: F401
from headsup.engine import HeadsUpPoker, OBS_DIM  # noqa: F401
