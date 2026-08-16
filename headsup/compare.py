"""Head-to-head comparison of policies (and simple bots).

    python -m headsup.compare cfr sdcfr:runs/x/iterates.pt cfr:models/deepcfr_policy_v1.pth --hands 400000
    python -m headsup.compare cfr onnx --hands 200000 --bots

Every pair plays ``--hands`` hands with alternating seats; the table shows the row player's
mean chips/hand against the column player with the standard error.  ``--bots`` appends the
usual random / call / all-in opponents.  Player specs: cfr[:path] | sdcfr:path[@exact] |
onnx[:path] | random | call | allin | raise.
"""

import argparse
import itertools
import json
import time

import numpy as np

from headsup.env import make_vec_env, play_hands
from headsup.players import make_player


def head_to_head(spec_a, spec_b, hands, num_envs=1024, seed=0, device=None):
    """Mean and standard error of A's chips/hand against B (both seats, alternating)."""
    a = make_player(spec_a, device=device, seed=seed)
    b = make_player(spec_b, device=device, seed=seed + 1)
    env = make_vec_env(num_envs, b, seed=seed)
    r = play_hands(env, a, hands)
    return float(r.mean()), float(r.std() / np.sqrt(len(r)))


def compare(specs, hands, num_envs=1024, seed=0, device=None, bots=False, progress=print):
    opponents = list(specs) + (["random", "call", "allin"] if bots else [])
    results = {}
    for i, j in itertools.product(range(len(specs)), range(len(opponents))):
        a, b = specs[i], opponents[j]
        if a == b:
            continue
        if b in specs and (b, a) in results:  # antisymmetric: reuse
            m, se = results[(b, a)]
            results[(a, b)] = (-m, se)
            continue
        t0 = time.perf_counter()
        results[(a, b)] = head_to_head(a, b, hands, num_envs, seed, device)
        m, se = results[(a, b)]
        progress(f"{a} vs {b}: {m:+.3f} ± {se:.3f} chips/hand  ({time.perf_counter() - t0:.0f}s)")
    return results, opponents


def format_table(specs, opponents, results):
    short = lambda s: s if len(s) <= 28 else "…" + s[-27:]
    w = max(len(short(s)) for s in specs + opponents) + 2
    head = " " * w + "".join(f"{short(o):>{w}}" for o in opponents)
    lines = [head]
    for a in specs:
        row = f"{short(a):<{w}}"
        for b in opponents:
            if a == b:
                row += f"{'—':>{w}}"
            else:
                m, se = results[(a, b)]
                row += f"{m:+.3f}±{se:.3f}".rjust(w)
        lines.append(row)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="two or more player specs")
    p.add_argument("--hands", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--bots", action="store_true", help="also play against random / call / all-in")
    p.add_argument("--json", default=None, help="write results to this file")
    args = p.parse_args()
    if len(args.specs) < 2 and not args.bots:
        p.error("give at least two specs (or one spec with --bots)")
    results, opponents = compare(args.specs, args.hands, args.num_envs, args.seed, args.device, args.bots)
    print("\nrow player's chips/hand vs column player (± standard error), 1 chip = 500 mbb\n")
    print(format_table(args.specs, opponents, results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({f"{a} | {b}": {"mean": m, "se": se} for (a, b), (m, se) in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
