"""AlphaHoldem trainer: Trinal-Clip PPO on K-Best self-play in the two-seat env.

    python -m headsup.alphaholdem.train --out runs/alpha --iterations 5000 --device cuda:0
    python -m headsup.alphaholdem.train --out runs/alpha --iterations 8000 --device cuda:0 --resume

One iteration:

1. Every table gets an opponent for the iteration - the current agent (both seats are then the main agent) or one
   of the pool's survivors (the main agent's seat alternates from hand to hand) -, uniformly.
2. All tables play in lock-step; the main network acts for its seats (sampled, masked softmax), every pool member
   for its tables.  After ``--samples`` main decisions the tables whose hand is in progress finish it while the
   others wait, so the rollout holds complete hands only.
3. GAE along the (table, seat) streams, value target = the hand's discounted return (``--value-clip``: clipped to
   the chips put in up to the state), ``--epochs`` passes of Trinal-Clip PPO in minibatches of ``--minibatch``.
4. The hands against each pool member are one ELO game; every ``--snapshot-every`` iterations a frozen copy of the
   main agent joins the pool, which keeps its ``--pool`` best.

``--out`` gets ``checkpoint.pt`` (everything needed to ``--resume``; a resumed run draws new cards, it is not a
bit-wise continuation), ``policy.pth`` (player spec ``alpha:<out>/policy.pth``) and ``log.json`` (one record per
iteration; every ``--eval-every`` iterations with the chips/hand ± SE against random / call / allin, all-in EV).
On ``--resume`` the hyperparameters are the checkpoint's; only the budget, evaluation and checkpoint flags apply
(a hyperparameter flag that says something else is reported and ignored).

What the paper leaves open and what was chosen: docs/superpowers/specs/2026-10-08-alphaholdem-design.md.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from headsup.alphaholdem.encoding import Encoder
from headsup.alphaholdem.model import AlphaNet
from headsup.alphaholdem.player import AlphaHoldemPlayer
from headsup.alphaholdem.pool import KBestPool
from headsup.alphaholdem.ppo import ppo_loss, stream_gae, value_fit
from headsup.game import DEFAULT_GAME, GameConfig
from headsup.twoseat import make_two_seat_env

DEFAULTS = dict(
    envs=4096,            # tables played in lock-step
    samples=131_072,      # main-agent decisions per iteration, at least (the paper: 8 x 128 envs x 128 steps)
    epochs=3,             # passes over a rollout (not in the paper)
    minibatch=16_384,     # the paper's total batch size
    lr=3e-4,              # Adam (paper: "initial learning rate 0.0003"; no schedule is described)
    gamma=0.999,          # paper
    lam=0.95,             # paper (GAE)
    eps=0.2,              # paper: "typical value"
    delta1=3.0,           # paper
    value_coef=0.5,       # not in the paper
    entropy_coef=0.01,    # not in the paper
    max_grad_norm=0.5,    # not in the paper
    reward_scale=0.0,     # chips per unit of reward; 0 = the stack size
    adv_norm=True,        # advantages normalised over the rollout (not in the paper)
    value_clip=False,     # clip the value target to per-state bounds (off: it biases the advantages, see ppo.py)
    pool=8,               # K (not in the paper)
    snapshot_every=50,    # iterations between snapshots (not in the paper)
    elo_k=16.0,           # ELO K-factor
    channels=64, conv_layers=3, hidden=256,  # network
    allin_ev=False,       # all-in hands rewarded with their expectation over the runouts (not in the paper)
    ev_samples=100,       # sampled runouts for it where there are more
    seed=0,
    backend="auto",       # two-seat env: auto / cpp / python
)
LOSS_KEYS = ("cards", "acts", "legal", "action", "logp", "adv", "ret", "own", "opp")
EVAL_OPPONENTS = ("random", "call", "allin")


class Trainer:
    def __init__(self, config=None, game=DEFAULT_GAME, device="cpu", **overrides):
        self.cfg = c = {**DEFAULTS, **(config or {}), **overrides}
        unknown = set(c) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        self.game, self.device = game, torch.device(device)
        torch.manual_seed(c["seed"])
        self.net = AlphaNet(game, c["channels"], c["conv_layers"], c["hidden"]).to(self.device).eval()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"], eps=1e-5)
        self.encoder = Encoder(game, self.device)
        self.pool = KBestPool(c["pool"], c["elo_k"])
        self.reward_scale = float(c["reward_scale"] or game.stack_size)
        self.rng = np.random.default_rng(c["seed"])
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(c["seed"])
        self.iteration = self.samples = self.hands = 0
        self.seconds = 0.0
        self.log = []
        self._make_env()

    def _make_env(self):
        c = self.cfg  # a resumed run must not replay the first run's cards: the seed moves with the iteration
        self.env = make_two_seat_env(c["envs"], seed=c["seed"] + 1_000_003 * self.iteration, backend=c["backend"], game=self.game)
        self.env.allin_ev, self.env.ev_samples = bool(c["allin_ev"]), int(c["ev_samples"])

    # ------------------------------------------------------------------ rollout
    def _act(self, net, cards, acts, legal):
        logits, value = net(cards, acts, legal)
        logp = torch.log_softmax(logits, dim=-1)
        action = torch.multinomial(logp.exp(), 1, generator=self.gen)  # masked actions have probability exactly 0
        return action.squeeze(1), logp.gather(1, action).squeeze(1), value

    @torch.no_grad()
    def collect(self, assignment=None, keep_obs=False):
        """Play one rollout.  Returns ``(batch, info)``: the main agent's decisions (network inputs, action,
        log-probability, value, value-clip bounds, advantage, return, and where they happened: step / table / seat)
        and the rollout's statistics, with the [step, table] grid of what every table did.  ``assignment``
        (int[envs]; -1 = the current agent, else a pool index) overrides the pool's opponent assignment and
        ``keep_obs`` adds the observations to the grid (tests)."""
        c, dev, n = self.cfg, self.device, self.cfg["envs"]
        opponent_np = np.asarray(self.pool.assign(n, self.rng) if assignment is None else assignment, dtype=np.int64)
        main_seat_np = self.rng.integers(0, 2, n)  # the main agent's seat at tables against a pool member
        opponent, main_seat = torch.tensor(opponent_np, device=dev), torch.tensor(main_seat_np, device=dev)  # copies
        pool_chips, pool_hands = np.zeros(len(self.pool)), np.zeros(len(self.pool), dtype=np.int64)
        store = {k: [] for k in ("cards", "acts", "legal", "action", "logp", "value", "own", "opp", "step", "table", "seat")}
        grid = {k: [] for k in ("seat", "active", "done", "reward", "value")}
        info = dict(opponent=opponent_np.copy(), main_seat=main_seat_np.copy())  # as the rollout starts
        observations = []
        fresh = torch.ones(n, dtype=torch.bool, device=dev)  # no action yet in the table's hand
        count, step, draining = 0, 0, False
        self.net.eval()
        obs, seat = self.env.reset()
        while True:
            active = ~fresh if draining else torch.ones_like(fresh)  # at the end only hands in progress continue
            if not active.any():
                break
            obs_t, seat_t = torch.as_tensor(obs, device=dev), torch.as_tensor(seat, device=dev)
            if keep_obs:
                observations.append(obs_t)
            cards, acts, legal = self.encoder(obs_t)
            main = active & ((opponent < 0) | (seat_t == main_seat))
            actions = torch.full((n,), -1, dtype=torch.long, device=dev)  # -1: the table waits
            values = torch.zeros(n, device=dev)
            rows = main.nonzero().squeeze(1)
            if len(rows):
                action, logp, value = self._act(self.net, cards[rows], acts[rows], legal[rows])
                actions[rows], values[rows] = action, value
                own, opp = self.encoder.value_bounds(obs_t[rows])
                for k, v in zip(store, (cards[rows], acts[rows], legal[rows], action, logp, value, own, opp,
                                        torch.full_like(rows, step), rows, seat_t[rows])):
                    store[k].append(v)
                count += len(rows)
            for j, member in enumerate(self.pool.members):
                rows = (active & ~main & (opponent == j)).nonzero().squeeze(1)
                if len(rows):
                    actions[rows] = self._act(member.net, cards[rows], acts[rows], legal[rows])[0]
            obs, seat, rewards, dones = self.env.step(actions.cpu().numpy())
            ended = np.flatnonzero(dones & (opponent_np >= 0))
            if len(ended):  # the result against the pool member, then the main agent changes its seat
                np.add.at(pool_chips, opponent_np[ended], rewards[ended, main_seat_np[ended]])
                np.add.at(pool_hands, opponent_np[ended], 1)
                main_seat_np[ended] ^= 1
                main_seat = torch.tensor(main_seat_np, device=dev)
            rewards, dones = torch.as_tensor(rewards, device=dev), torch.as_tensor(dones, device=dev)
            for k, v in zip(grid, (seat_t, active, dones, rewards, values)):
                grid[k].append(v)
            fresh = torch.where(active, dones, fresh)
            step += 1
            draining = draining or count >= c["samples"]
        grid = {k: torch.stack(v) for k, v in grid.items()}
        adv, ret = stream_gae(grid["value"], grid["reward"] / self.reward_scale, grid["done"], grid["seat"], grid["active"],
                              c["gamma"], c["lam"])
        batch = {k: torch.cat(v) for k, v in store.items()}
        batch["adv"], batch["ret"] = adv[batch["step"], batch["table"]], ret[batch["step"], batch["table"]]
        if keep_obs:
            grid["obs"] = torch.stack(observations)
        info.update(steps=step, hands=int(grid["done"].sum()), pool_chips=pool_chips, pool_hands=pool_hands, grid=grid)
        return batch, info

    # ------------------------------------------------------------------ update
    def update(self, batch):
        """``epochs`` passes of Trinal-Clip PPO over the rollout; returns the mean statistics of the minibatches."""
        c, size = self.cfg, len(batch["action"])
        batch = {k: batch[k] for k in LOSS_KEYS}
        if c["adv_norm"]:
            batch["adv"] = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        minibatches = max(1, round(size / c["minibatch"]))
        totals, steps = {}, 0
        self.net.train()
        for _ in range(c["epochs"]):
            for idx in torch.randperm(size, generator=self.gen, device=self.device).tensor_split(minibatches):
                loss, stats = ppo_loss(self.net, {k: v[idx] for k, v in batch.items()}, eps=c["eps"], delta1=c["delta1"],
                                       value_coef=c["value_coef"], entropy_coef=c["entropy_coef"], reward_scale=self.reward_scale,
                                       value_clip=c["value_clip"])
                stats["loss"] = loss.item()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), c["max_grad_norm"]))
                self.opt.step()
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0.0) + v
                steps += 1
        self.net.eval()
        return {k: v / steps for k, v in totals.items()}

    def iterate(self):
        """One iteration (rollout, update, ELO games, snapshot); returns its log record."""
        c, t0 = self.cfg, time.perf_counter()
        batch, info = self.collect()
        t1 = time.perf_counter()
        size = len(batch["action"])
        fit = value_fit(batch["ret"], batch["value"], batch["own"], batch["opp"], self.reward_scale, c["value_clip"])
        actions = torch.bincount(batch["action"], minlength=self.game.num_actions).float() / size
        stats = self.update(batch)
        self.iteration += 1
        self.samples += size
        self.hands += info["hands"]
        for j in range(len(self.pool)):
            self.pool.record(j, float(info["pool_chips"][j]), int(info["pool_hands"][j]))
        vs_hands = int(info["pool_hands"].sum())
        vs_pool = [[m.iteration, float(chips / max(hands, 1)), int(hands)]  # per member: the main agent's chips / hand
                   for m, chips, hands in zip(self.pool.members, info["pool_chips"], info["pool_hands"])]
        if self.iteration % c["snapshot_every"] == 0:
            self.pool.add(self.net, self.iteration)
        seconds = time.perf_counter() - t0
        self.seconds += seconds
        record = dict(
            iteration=self.iteration, samples=self.samples, hands=self.hands, batch=size, steps=info["steps"],
            seconds=self.seconds, rollout_seconds=t1 - t0, samples_per_second=size / seconds, **stats,
            **fit, actions=actions.tolist(), elo=self.pool.main_elo,
            pool=[[m.iteration, m.elo] for m in self.pool.members], vs_pool=vs_pool, hands_vs_pool=vs_hands,
            chips_vs_pool=float(info["pool_chips"].sum() / vs_hands) if vs_hands else 0.0,
        )
        self.log.append(record)
        return record

    # ------------------------------------------------------------------ evaluation
    def player(self, deterministic=False, seed=0):
        return AlphaHoldemPlayer(self.net, deterministic=deterministic, seed=seed)

    def evaluate(self, hands, opponents=EVAL_OPPONENTS, num_envs=1024, seed=0):
        """{opponent: [chips/hand, standard error]} of the current agent (sampled actions; all-in EV rewards).
        ``cfr`` is the shipped DeepCFR network, run inside the C++ env."""
        from headsup.env import make_vec_env, play_hands

        player, out = self.player(seed=seed), {}
        for i, name in enumerate(opponents):
            opponent = name
            if name == "cfr":
                from headsup.model import load_model
                from headsup.paths import DEFAULT_POLICY_PATH

                opponent = load_model(DEFAULT_POLICY_PATH, device="cpu")
            env = make_vec_env(min(num_envs, hands), opponent, seed=seed + i, game=self.game)
            r = play_hands(env, player, hands, allin_ev=True)
            out[name] = [float(r.mean()), float(r.std() / np.sqrt(len(r)))]
        return out

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self):
        return dict(config=dict(self.cfg), game=self.game.to_dict(), net={k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                    opt=self.opt.state_dict(), pool=self.pool.state_dict(), iteration=self.iteration, samples=self.samples,
                    hands=self.hands, seconds=self.seconds, log=self.log, rng=self.rng.bit_generator.state, gen=self._generator_state())

    def _generator_state(self):
        try:
            return self.gen.get_state()
        except RuntimeError:  # a device whose generator cannot be saved: the resumed run starts a fresh stream
            return None

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
        self.opt.load_state_dict(state["opt"])
        self.pool.load_state_dict(state["pool"], device=self.device)
        self.iteration, self.samples, self.hands = state["iteration"], state["samples"], state["hands"]
        self.seconds, self.log = state["seconds"], list(state["log"])
        self.rng.bit_generator.state = state["rng"]
        try:
            self.gen.set_state(state["gen"])
        except (RuntimeError, TypeError):  # no saved state, or one of another device type: a fresh stream
            self.gen.manual_seed(self.cfg["seed"] + self.iteration)
        self._make_env()

    def save(self, out):
        """Write ``checkpoint.pt``, ``policy.pth`` and ``log.json`` into ``out`` (each atomically)."""
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "checkpoint.pt")
        torch.save(self.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)
        path = os.path.join(out, "policy.pth")
        self.net.save(path + ".tmp")
        os.replace(path + ".tmp", path)
        path = os.path.join(out, "log.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"config": self.cfg, "game": self.game.tree_dict(), "parameters": self.net.parameter_counts(), "log": self.log}, f, indent=1)
        os.replace(path + ".tmp", path)

    @staticmethod
    def resume(out, device="cpu"):
        state = torch.load(os.path.join(out, "checkpoint.pt"), map_location="cpu", weights_only=False)
        trainer = Trainer(state["config"], game=GameConfig.from_dict(state["game"]), device=device)
        trainer.load_state_dict(state)
        return trainer


def _line(r):
    pool = f"pool {len(r['pool'])} elo {r['elo']:.0f} vs pool {r['chips_vs_pool']:+.2f}" if r["pool"] else "pool 0"
    return (f"it {r['iteration']:5d}  samples {r['samples'] / 1e6:8.2f}M  {r['samples_per_second'] / 1e3:6.1f}k/s  policy {r['policy']:+.4f}  "
            f"value {r['value']:.4f}  entropy {r['entropy']:.3f}  kl {r['kl']:.4f}  clipped {r['clipped']:.2f}/{r['delta1_clipped']:.3f}/"
            f"{r['value_clipped']:.2f}  actions {' '.join(f'{a:.2f}' for a in r['actions'])}  {pool}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("--out", required=True, help="run directory (checkpoint.pt, policy.pth, log.json)")
    p.add_argument("--iterations", type=int, required=True, help="train until this iteration")
    p.add_argument("--device", default=None, help="torch device (default: auto)")
    p.add_argument("--resume", action="store_true", help="continue the run in --out (its hyperparameters are kept)")
    d = DEFAULTS
    p.add_argument("--envs", type=int, default=d["envs"], help="tables played in lock-step")
    p.add_argument("--samples", type=int, default=d["samples"], help="main-agent decisions per iteration (at least; hands are completed)")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--minibatch", type=int, default=d["minibatch"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--gamma", type=float, default=d["gamma"])
    p.add_argument("--lam", type=float, default=d["lam"], help="GAE lambda")
    p.add_argument("--eps", type=float, default=d["eps"], help="PPO clip")
    p.add_argument("--delta1", type=float, default=d["delta1"], help="upper ratio clip for negative advantages")
    p.add_argument("--value-coef", type=float, default=d["value_coef"])
    p.add_argument("--entropy-coef", type=float, default=d["entropy_coef"])
    p.add_argument("--max-grad-norm", type=float, default=d["max_grad_norm"])
    p.add_argument("--reward-scale", type=float, default=d["reward_scale"], help="chips per unit of reward (0: the stack size)")
    p.add_argument("--no-adv-norm", dest="adv_norm", action="store_false", help="do not normalise advantages over the rollout")
    p.add_argument("--value-clip", dest="value_clip", action="store_true",
                   help="clip the value target to [-own chips, opponent's chips] at the state (the per-state reading of delta2 / delta3)")
    p.add_argument("--no-value-clip", dest="value_clip", action="store_false", help="the hand's return as value target (the default)")
    p.set_defaults(value_clip=d["value_clip"])
    p.add_argument("--pool", type=int, default=d["pool"], help="K: survivors kept in the pool")
    p.add_argument("--snapshot-every", type=int, default=d["snapshot_every"], help="iterations between snapshots of the main agent")
    p.add_argument("--elo-k", type=float, default=d["elo_k"])
    p.add_argument("--channels", type=int, default=d["channels"])
    p.add_argument("--conv-layers", type=int, default=d["conv_layers"])
    p.add_argument("--hidden", type=int, default=d["hidden"])
    p.add_argument("--allin-ev", action="store_true", help="reward all-in hands with their expectation over the runouts (not in the paper)")
    p.add_argument("--ev-samples", type=int, default=d["ev_samples"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--backend", default=d["backend"], choices=["auto", "cpp", "python"], help="two-seat env implementation")
    p.add_argument("--eval-every", type=int, default=50, help="iterations between evaluations (0: never; the last iteration always)")
    p.add_argument("--eval-hands", type=int, default=20_000)
    p.add_argument("--eval-cfr", action="store_true", help="also evaluate against the shipped DeepCFR network")
    p.add_argument("--checkpoint-every", type=int, default=25, help="iterations between checkpoints (the last iteration always)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch's default)")
    return p


def overridden_flags(parser, argv, args, config):
    """The hyperparameter options of ``argv`` whose value differs from ``config`` (a checkpoint's settings, which a
    resumed run keeps), as text for a warning."""
    out = []
    for action in parser._actions:
        given = [opt for opt in action.option_strings if any(a == opt or a.startswith(opt + "=") for a in argv)]
        if given and action.dest in config and getattr(args, action.dest) != config[action.dest]:
            out.append(f"{given[0]} (the checkpoint has {action.dest} = {config[action.dest]!r})")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    p = build_parser()
    args = p.parse_args(argv)

    from headsup.device import get_device

    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    checkpoint = os.path.join(args.out, "checkpoint.pt")
    if args.resume:
        if not os.path.exists(checkpoint):
            p.error(f"--resume: no checkpoint in {args.out}")
        trainer = Trainer.resume(args.out, device=device)
        print(f"resumed {args.out} at iteration {trainer.iteration} ({trainer.samples:,} samples)", flush=True)
        ignored = overridden_flags(p, argv, args, trainer.cfg)
        if ignored:
            print("warning: a resumed run keeps its checkpoint's hyperparameters; ignored: " + ", ".join(ignored), flush=True)
    else:
        if os.path.exists(checkpoint):
            p.error(f"{args.out} already holds a run: continue it with --resume or choose another --out")
        trainer = Trainer({k: getattr(args, k) for k in DEFAULTS}, device=device)
    counts = trainer.net.parameter_counts()
    print(f"AlphaHoldem on {device}: {counts['total']:,} parameters ({counts['conv']:,} conv, {counts['fc']:,} fc), "
          f"{trainer.cfg['envs']} tables, >= {trainer.cfg['samples']:,} samples / iteration", flush=True)
    opponents = EVAL_OPPONENTS + (("cfr",) if args.eval_cfr else ())
    while trainer.iteration < args.iterations:
        record = trainer.iterate()
        last = trainer.iteration == args.iterations
        print(_line(record), flush=True)
        if args.eval_hands and (last or (args.eval_every and trainer.iteration % args.eval_every == 0)):
            record["eval"] = trainer.evaluate(args.eval_hands, opponents, seed=trainer.iteration)
            print(f"eval it {trainer.iteration}: " + "  ".join(f"{k} {m:+.3f} ± {se:.3f}" for k, (m, se) in record["eval"].items()), flush=True)
        if last or (args.checkpoint_every and trainer.iteration % args.checkpoint_every == 0):
            trainer.save(args.out)
    if not os.path.exists(os.path.join(args.out, "policy.pth")):
        trainer.save(args.out)
    return trainer


if __name__ == "__main__":
    main(sys.argv[1:])
