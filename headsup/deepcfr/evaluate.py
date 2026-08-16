"""Evaluate a policy against simple opponents (and optionally an exploiter) in chips/hand.

    python -m headsup.deepcfr.evaluate --policy models/deepcfr_policy.pth --hands 100000
"""

import argparse
import time

import numpy as np

from headsup.env import make_vec_env, play_hands

DEFAULT_OPPONENTS = ("random", "call", "allin")


def evaluate(player, hands, opponents=DEFAULT_OPPONENTS, num_envs=1024, seed=0, backend="auto", progress=False):
    """``player`` is a batched player; returns {opponent: mean chips per hand}.  Envs use the
    player's action tree (``player.game``) when it has one."""
    scores = {}
    game = getattr(player, "game", None)
    for i, opp in enumerate(opponents):
        env = make_vec_env(num_envs, opp, seed=seed + i, backend=backend, game=game)
        rewards = play_hands(env, player, hands, progress=progress)
        scores[opp] = float(rewards.mean())
    return scores


def evaluate_model(model, device, hands, opponents=DEFAULT_OPPONENTS, num_envs=1024, seed=0, deterministic=False):
    from headsup.players import TorchPolicyPlayer

    return evaluate(TorchPolicyPlayer(model, device=device, deterministic=deterministic, seed=seed), hands, opponents, num_envs, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="cfr", help="player spec: cfr[:path.pth] | onnx[:path.onnx] | random | call | allin")
    parser.add_argument("--opponents", default="random,call,allin", help="comma separated player specs")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--device", default=None, help="torch device for the evaluated policy (default: auto)")
    parser.add_argument("--deterministic", action="store_true", help="argmax instead of sampling for the evaluated policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from headsup.players import make_player

    player = make_player(args.policy, device=args.device, deterministic=args.deterministic, seed=args.seed)
    for opp in args.opponents.split(","):
        env = make_vec_env(args.num_envs, opp.strip(), seed=args.seed, game=getattr(player, "game", None))
        t0 = time.perf_counter()
        r = play_hands(env, player, args.hands, progress=True)
        dt = time.perf_counter() - t0
        se = r.std() / np.sqrt(len(r))
        print(f"{args.policy} vs {opp:>10s}: {r.mean():+.3f} ± {se:.3f} chips/hand  ({r.mean()*500:+.0f} mbb/hand)  [{len(r)/dt:,.0f} hands/s]")


if __name__ == "__main__":
    main()
