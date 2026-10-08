"""AlphaHoldem (Zhao, Yan, Li, Li, Xing, AAAI-22) for the default no-limit game: end-to-end self-play RL.

- ``encoding``: the card / action tensors, rebuilt from observations.
- ``model``: the pseudo-siamese network (one ConvNet per tensor, policy and value heads).
- ``ppo``: Trinal-Clip PPO loss, advantages along (table, seat) streams.
- ``pool``: K-Best self-play pool with ELO.
- ``train``: the trainer (``python -m headsup.alphaholdem.train --out runs/x``).
- ``player``: the stateless player behind the spec ``alpha:<path>``.

Design, the paper's open points and the choices made: docs/superpowers/specs/2026-10-08-alphaholdem-design.md
"""
