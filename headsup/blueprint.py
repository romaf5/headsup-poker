"""Tabular blueprint strategies: Pluribus's MCCFR-P over the public betting tree x a card
abstraction (Brown & Sandholm 2019, supplementary material, Algorithm 1), C++ (`headsup_cpp.TabularBlueprint`).

Infosets = (public betting sequence, card bucket): 169 lossless hand classes pre-flop, `buckets`
equal-mass buckets of the expected hand strength (equity vs a uniform random hand; Monte-Carlo
runouts on the flop / turn, exact on the river) on the later rounds - equal-width in the *mass*
of situations rather than Pluribus's k-means over equity distributions.  Training: external-
sampling MCCFR with unweighted regret updates and periodic linear discounting (Linear MCCFR),
negative-regret pruning of the traverser's actions in 95 % of iterations after a warm-up (never
on the last round or into terminals), a regret floor, the average strategy from sampled action
counters (UPDATE-STRATEGY every 10 000 iterations - on every round here, Pluribus only tracks
the first round and averages later-round snapshots).  Iterations play the role of Pluribus's
minutes.  Threads share the tables (benign races).

    python -m headsup.blueprint --game nlhe --iterations 10000000 --threads 32 --out runs/bp/blueprint.pt
    python -m headsup.compare tab:runs/bp/blueprint.pt cfr:runs/x/policy.pth --hands 200000
    python -m headsup.web --opponent search:tab:runs/bp/blueprint.pt@pluribus@th16   # Pluribus: blueprint + search

Player spec ``tab:path.pt[@current]`` (``@current``: regret-matching current strategy instead of
the average).  Observations are mapped to public nodes by replaying the bet history they carry
(`headsup.public.replay_from_obs`), rows sharing a public state are grouped (LBR / search hand
substitution queries 1326 rows of one state at once).
"""

import argparse
import json
import time

import numpy as np

from headsup import native
from headsup.game import DEFAULT_GAME, GameConfig
from headsup.public import board_cards, hero_cards, replay_from_obs


class TabularBlueprint:
    """The C++ trainer/policy plus its parameters; ``save`` / ``load`` round-trip everything."""

    def __init__(self, game=DEFAULT_GAME, buckets=200, samples=500):
        self.game = game
        self.cpp = native.module().TabularBlueprint()
        self.cpp.build(native.engine_config(game=game), buckets, samples)
        self.params = dict(prune_threshold=-300.0 * game.stack_size * 100, regret_floor=-310.0 * game.stack_size * 100,
                           prune_after=0, lcfr_iterations=0, discount_interval=0, strategy_interval=10000, prune_prob=0.95,
                           dense_average=True)
        self._apply_params()

    def _apply_params(self):
        p = self.params
        self.cpp.set_params(p["prune_threshold"], p["regret_floor"], int(p["prune_after"]), int(p["lcfr_iterations"]),
                            int(p["discount_interval"]), int(p["strategy_interval"]), p["prune_prob"], bool(p.get("dense_average", True)))

    def configure(self, **params):
        """Pluribus's schedule in iterations: ``lcfr_iterations`` (discount for that long, every
        ``discount_interval``), ``prune_after`` (start pruning), thresholds in chips (defaults scale
        Pluribus's -300M / -310M for 10 000-chip stacks to the game's stack)."""
        self.params.update(params)
        self._apply_params()
        return self

    def fit_abstraction(self, situations=200_000, seed=0, threads=16):
        self.cpp.fit_abstraction(situations, seed, threads)
        return self

    def run(self, iterations, seed=0, threads=16):
        self.cpp.run(int(iterations), int(seed), int(threads))
        return self

    @property
    def iterations(self):
        return int(self.cpp.iterations)

    def save(self, path):
        import torch

        torch.save({"kind": "tabular_blueprint", "game": self.game.to_dict(), "buckets": self.cpp.buckets, "samples": self.cpp.samples,
                    "edges": [np.asarray(e, dtype=np.float32) for e in self.cpp.edges], "regret": np.asarray(self.cpp.regret),
                    "phi": np.asarray(self.cpp.phi), "iterations": self.iterations, "params": self.params}, path)

    @classmethod
    def load(cls, path):
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("kind") != "tabular_blueprint":
            raise ValueError(f"{path} is not a tabular blueprint")
        bp = cls(GameConfig.from_dict(data["game"]), data["buckets"], data["samples"])
        bp.params.update(data["params"])
        bp._apply_params()
        bp.cpp.edges = [list(map(float, e)) for e in data["edges"]]
        bp.cpp.regret = np.asarray(data["regret"], dtype=np.float32)
        bp.cpp.phi = np.asarray(data["phi"], dtype=np.float32)
        bp.cpp.iterations = int(data["iterations"])
        return bp

    # -- queries ---------------------------------------------------------------------------
    def node_of(self, actions):
        node = 0
        for a in actions:
            node = self.cpp.child(node, int(a))
            if node < 0:
                raise ValueError("action sequence leaves the tree")
        return node

    def strategy_for_hands(self, node, hands, board, seed=0, current=False):
        return np.asarray(self.cpp.strategy_for_hands(int(node), np.asarray(hands, dtype=np.int32).reshape(-1, 2), list(map(int, board)), int(seed), current))


class TabularPlayer:
    """Player protocol on top of a :class:`TabularBlueprint` (``player(obs) -> actions``, ``probs``)."""

    wants_ids = False

    def __init__(self, blueprint, current=False, seed=None):
        self.bp = blueprint if isinstance(blueprint, TabularBlueprint) else TabularBlueprint.load(blueprint)
        self.game = self.bp.game
        self.current = current
        self.rng = np.random.default_rng(seed)
        self.last_probs = None
        self._nodes = {}  # public-history key -> node

    def _node(self, row):
        key = row[21:].tobytes()  # stage, position, pot features, history: identifies the public state
        node = self._nodes.get(key)
        if node is None:
            actions = []
            replay_from_obs(row, self.game, on_action=lambda e, seat, a: actions.append(int(a)))
            node = self._nodes[key] = self.bp.node_of(actions)
        return node

    def probs(self, obs, ids=None):
        from headsup.engine import legal_mask_from_obs

        obs = np.asarray(obs, dtype=np.float32)
        out = np.zeros((len(obs), self.game.num_actions), dtype=np.float32)
        # rows sharing the public part (everything but the hand slots) form one query
        _, first, inverse = np.unique(obs[:, 6:], axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        all_hands = np.sort(np.stack([obs[:, 2], obs[:, 5]], axis=1).astype(np.int32) - 1, axis=1)
        for g, ref_i in enumerate(first):
            idx = np.flatnonzero(inverse == g)
            row = obs[ref_i]
            board = board_cards(row)
            hands = all_hands[idx]
            # a valid hero hand is needed to replay the public state (hand-substituted rows may overlap the board)
            valid = ~np.isin(hands, board).any(axis=1)
            ref = obs[idx[np.argmax(valid)]] if valid.any() else row
            try:
                node = self._node(ref)
            except ValueError:  # terminal observation (the hand is over): any action
                out[idx, 1] = 1.0
                continue
            if self.bp.cpp.node_player(node) < 0:
                out[idx, 1] = 1.0
                continue
            out[idx] = self.bp.strategy_for_hands(node, hands, board, int(self.rng.integers(2**31)), self.current)
        legal = legal_mask_from_obs(obs, self.game)
        out[~legal] = 0.0
        s = out.sum(axis=1, keepdims=True)
        return np.where(s > 0, out / np.maximum(s, 1e-12), legal / legal.sum(axis=1, keepdims=True)).astype(np.float32)

    def __call__(self, obs, ids=None):
        from headsup.players import sample_actions

        self.last_probs = self.probs(obs, ids)
        return sample_actions(self.last_probs, self.rng)


def parse_tab_spec(arg):
    parts = arg.split("@")
    return parts[0], "current" in parts[1:]


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    from headsup.deepcfr.evaluate import evaluate
    from headsup.games.holdem import make_holdem

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="nlhe", choices=["nlhe", "fhp", "hulh"])
    p.add_argument("--bet-sizes", default=None, help="NL raise sizes, e.g. 'min' or '0.5,1,2' (pot fractions)")
    p.add_argument("--iterations", type=int, default=10_000_000, help="MCCFR iterations (each = one traversal per seat)")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--buckets", type=int, default=200)
    p.add_argument("--samples", type=int, default=500, help="Monte-Carlo runouts per EHS evaluation (flop / turn)")
    p.add_argument("--situations", type=int, default=200_000, help="random situations per round to fit the bucket edges")
    p.add_argument("--lcfr", type=float, default=0.4, help="fraction of the iterations with linear discounting (Pluribus: 400 of 800+ minutes)")
    p.add_argument("--discount-every", type=float, default=0.01, help="discount interval as a fraction of the iterations (Pluribus: 10 of 400 minutes)")
    p.add_argument("--prune-after", type=float, default=0.2, help="start pruning after this fraction of the iterations (Pluribus: 200 minutes)")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--strategy-every", type=int, default=None,
                   help="UPDATE-STRATEGY interval in iterations (Pluribus: 10 000 of ~1e9+; default: iterations / 1 000 000, at least 1 - "
                        "the sampled counters need many visits per infoset to be a precise average)")
    p.add_argument("--chunks", type=int, default=20, help="training chunks (checkpoint + evaluation after each)")
    p.add_argument("--eval-hands", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output file (.pt)")
    args = p.parse_args(argv)
    from headsup.game import parse_bet_sizes

    game = make_holdem(args.game, **({"bet_sizes": parse_bet_sizes(args.bet_sizes)} if args.bet_sizes else {}))
    bp = TabularBlueprint(game, args.buckets, args.samples)
    bp.configure(lcfr_iterations=int(args.lcfr * args.iterations), discount_interval=max(1, int(args.discount_every * args.iterations)),
                 prune_after=0 if args.no_prune else int(args.prune_after * args.iterations),
                 strategy_interval=args.strategy_every or max(1, args.iterations // 1_000_000))
    print(f"{args.game}: {bp.cpp.num_nodes} public nodes, {bp.cpp.num_infosets:,} infosets; params {bp.params}", flush=True)
    t0 = time.perf_counter()
    bp.fit_abstraction(args.situations, args.seed, args.threads)
    print(f"abstraction fitted in {time.perf_counter() - t0:.1f}s", flush=True)
    log = []
    per_chunk = args.iterations // args.chunks
    for c in range(args.chunks):
        t0 = time.perf_counter()
        bp.run(per_chunk, args.seed + 1000 * (c + 1), args.threads)
        dt = time.perf_counter() - t0
        bp.save(args.out)
        entry = {"iterations": bp.iterations, "seconds": dt}
        if args.eval_hands:
            scores = evaluate(TabularPlayer(bp, seed=c), args.eval_hands, seed=c)
            entry.update(scores)
        log.append(entry)
        print(f"chunk {c + 1}/{args.chunks}: {bp.iterations:,} iterations ({per_chunk / dt:,.0f} it/s) " +
              " ".join(f"vs {k} {v:+.2f}" for k, v in entry.items() if k not in ("iterations", "seconds")), flush=True)
        with open(str(args.out) + ".log.json", "w") as f:
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
