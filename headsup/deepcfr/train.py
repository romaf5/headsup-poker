"""DeepCFR training (Brown et al. 2019) for heads-up hold'em, tuned for a Mac (MPS) but
device-agnostic.

    python -m headsup.deepcfr.train --iterations 300 --traversals 10000 --out runs/deepcfr

Each CFR iteration, for each seat:
  1. traverse the game ``--traversals`` times with external sampling (C++ kernel, all cores),
  2. add the traverser's advantage samples to its reservoir memory and the opponent's
     strategies to the strategy memory,
  3. train the seat's advantage network from scratch on its memory (GPU/MPS).
At the end (or on Ctrl-C) the average-strategy (policy) network is trained on the strategy
memory, saved to ``<out>/policy.pth`` and evaluated against simple opponents.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from headsup.deepcfr.evaluate import evaluate_model
from headsup.deepcfr.memory import ReservoirBuffer
from headsup.deepcfr.traverse import TraversalRunner
from headsup.device import get_device, synchronize
from headsup.model import BaseModel


# ----------------------------------------------------------------------------- networks
def maybe_compile(model, enabled=True):
    """torch.compile roughly halves the step time on MPS; fall back to eager if it fails."""
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model)
        return compiled
    except Exception as exc:  # pragma: no cover
        print(f"torch.compile unavailable ({type(exc).__name__}: {exc}); using eager mode")
        return model


def _run_step(fwd, model, opt, obs, t, target, loss_fn, grad_clip):
    pred = fwd(obs)
    loss = loss_fn(pred, t, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return loss


def _weighted_mse(pred, t, target):
    return (t[:, None] * (pred - target).pow(2)).mean()


def train_advantage_net(
    buffer, device, steps, batch_size, lr=1e-3, grad_clip=1.0, log=None, tag="", compile=True, log_step_offset=0, log_every=100
):
    """Fresh network fitted to (obs -> regrets) with iteration-weighted MSE.

    Returns ``(model, final_loss)``.  Per-step losses are logged at ``log_step_offset + step``
    so successive fits form one continuous curve in TensorBoard.
    """
    model = BaseModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    fwd = maybe_compile(model, compile)
    loss = None
    for step in range(steps):
        obs, t, target = buffer.sample(batch_size)
        try:
            loss = _run_step(fwd, model, opt, obs, t, target, _weighted_mse, grad_clip)
        except Exception as exc:
            if fwd is model:
                raise
            print(f"compiled step failed ({type(exc).__name__}); falling back to eager")
            fwd = model
            loss = _run_step(fwd, model, opt, obs, t, target, _weighted_mse, grad_clip)
        if log is not None and step % log_every == 0:
            log(f"advantage{tag}/loss", loss.item(), log_step_offset + step)
    model.eval()
    final_loss = float(loss.item()) if loss is not None else float("nan")
    if log is not None and steps:
        log(f"advantage{tag}/loss", final_loss, log_step_offset + steps - 1)
    return model, final_loss


def _policy_loss(logits, t, target):
    probs = torch.softmax(logits, dim=-1)
    return (t[:, None] * (probs - target).pow(2)).mean()


def train_policy_net(buffer, device, epochs, batch_size, lr=1e-3, gamma=0.9, log=None, progress=True, compile=True):
    """Average-strategy network: softmax(logits) fitted to stored strategies (weighted MSE)."""
    model = BaseModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(buffer) // batch_size)
    steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=steps_per_epoch, gamma=gamma)
    model.train()
    fwd = maybe_compile(model, compile)
    it = tqdm(range(steps), desc="policy", disable=not progress, leave=False)
    for step in it:
        obs, t, target = buffer.sample(batch_size)
        try:
            loss = _run_step(fwd, model, opt, obs, t, target, _policy_loss, None)
        except Exception as exc:
            if fwd is model:
                raise
            print(f"compiled step failed ({type(exc).__name__}); falling back to eager")
            fwd = model
            loss = _run_step(fwd, model, opt, obs, t, target, _policy_loss, None)
        scheduler.step()
        if step % 100 == 0 or step == steps - 1:
            value = loss.item()
            it.set_postfix(loss=f"{value:.5f}")
            if log is not None:
                log("policy/loss", value, step)
    model.eval()
    return model


# ----------------------------------------------------------------------------- trainer
class DeepCFRTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        os.makedirs(args.out, exist_ok=True)
        self.writer = None
        if not args.no_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(os.path.join(args.out, "tb"))
        self.iteration = 0
        self.seed_seq = np.random.SeedSequence(args.seed)
        torch.manual_seed(args.seed)

        self.adv_memory = [ReservoirBuffer(args.adv_capacity, self.device, seed=args.seed + i) for i in range(2)]
        self.strat_memory = ReservoirBuffer(args.strat_capacity, self.device, seed=args.seed + 2)
        self.nets = [BaseModel().to(self.device).eval() for _ in range(2)]
        self.runner = TraversalRunner(args.workers, backend=args.backend)
        print(
            f"device={self.device}  traversal backend={self.runner.backend} x{self.runner.num_workers} workers  "
            f"memories: adv {args.adv_capacity:,} x2, strat {args.strat_capacity:,}"
        )

    # -- logging ---------------------------------------------------------------------
    def log(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    # -- checkpointing ---------------------------------------------------------------
    def save_checkpoint(self, path=None):
        path = path or os.path.join(self.args.out, "checkpoint.pt")
        state = {
            "iteration": self.iteration,
            "nets": [n.state_dict() for n in self.nets],
            "adv_memory": [m.state_dict() for m in self.adv_memory],
            "strat_memory": self.strat_memory.state_dict(),
            "args": vars(self.args),
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.iteration = int(state["iteration"])
        for n, sd in zip(self.nets, state["nets"]):
            n.load_state_dict(sd)
        for m, sd in zip(self.adv_memory, state["adv_memory"]):
            m.load_state_dict(sd)
        self.strat_memory.load_state_dict(state["strat_memory"])
        print(f"resumed from {path} at iteration {self.iteration}")

    # -- one CFR iteration -----------------------------------------------------------
    def cfr_iteration(self):
        a = self.args
        self.iteration += 1
        t = float(self.iteration)  # linear CFR weight
        weights = [n.numpy_weights() for n in self.nets]
        for seat in range(2):
            t0 = time.perf_counter()
            seed = int(self.seed_seq.spawn(1)[0].generate_state(1)[0])
            adv, strat, nodes = self.runner.collect(weights, seat, a.traversals, t, seed)
            t_trav = time.perf_counter() - t0
            self.adv_memory[seat].add(adv.obs, adv.t, adv.target)
            self.strat_memory.add(strat.obs, strat.t, strat.target)

            t0 = time.perf_counter()
            self.nets[seat], final_loss = train_advantage_net(
                self.adv_memory[seat],
                self.device,
                a.value_steps,
                a.batch_size,
                a.lr,
                log=self.log,
                tag=f"/seat{seat}",
                compile=not a.no_compile,
                log_step_offset=(self.iteration - 1) * a.value_steps,
            )
            weights[seat] = self.nets[seat].numpy_weights()
            synchronize(self.device)
            t_train = time.perf_counter() - t0

            it = self.iteration
            self.log(f"advantage/seat{seat}/final_loss", final_loss, it)
            mse, target_rms = self._advantage_fit_quality(seat)
            self.log(f"advantage/seat{seat}/mse_unweighted", mse, it)
            self.log(f"advantage/seat{seat}/target_rms", target_rms, it)
            self.log(f"time/traverse/seat{seat}", t_trav, it)
            self.log(f"time/train_advantage/seat{seat}", t_train, it)
            self.log(f"samples/adv_per_traversal/seat{seat}", len(adv) / a.traversals, it)
            self.log(f"samples/strat_per_traversal/seat{seat}", len(strat) / a.traversals, it)
            self.log(f"samples/nodes_per_traversal/seat{seat}", nodes / a.traversals, it)
            self.log(f"memory/adv/seat{seat}", len(self.adv_memory[seat]), it)
            self.log("memory/strat", len(self.strat_memory), it)
            self.last_stats = dict(trav=t_trav, train=t_train, nodes=nodes / a.traversals, adv=len(adv), strat=len(strat))

    def _advantage_fit_quality(self, seat, n=65536):
        """Unweighted MSE of the freshly fitted net on a memory sample, and the target scale."""
        obs, _, target = self.adv_memory[seat].sample(min(n, len(self.adv_memory[seat])))
        with torch.no_grad():
            pred = self.nets[seat](obs)
            mse = (pred - target).pow(2).mean().item()
            rms = target.pow(2).mean().sqrt().item()
        return mse, rms

    def evaluate_iterate(self):
        """Current strategy (regret matching on the advantage nets) vs simple bots."""
        from headsup.deepcfr.evaluate import evaluate
        from headsup.players import RegretMatchingPlayer

        player = RegretMatchingPlayer(self.nets, device=self.device, seed=self.args.seed + self.iteration)
        scores = evaluate(player, self.args.iterate_eval_hands, num_envs=1024, seed=self.args.seed)
        for k, v in scores.items():
            self.log(f"eval_current_strategy/{k}", v, self.iteration)
        return scores

    def evaluate_quick_policy(self):
        """Quick average-strategy fit (few epochs) vs simple bots: the signal CFR actually improves."""
        a = self.args
        policy = train_policy_net(
            self.strat_memory, self.device, a.policy_eval_epochs, a.batch_size, a.lr, compile=not a.no_compile, progress=False
        )
        scores = evaluate_model(policy, self.device, a.iterate_eval_hands, seed=a.seed)
        for k, v in scores.items():
            self.log(f"eval_avg_strategy/{k}", v, self.iteration)
        return scores

    # -- policy ----------------------------------------------------------------------
    def train_policy(self):
        a = self.args
        t0 = time.perf_counter()
        policy = train_policy_net(
            self.strat_memory, self.device, a.policy_epochs, a.batch_size, a.lr, log=self.log, compile=not a.no_compile
        )
        synchronize(self.device)
        path = os.path.join(a.out, "policy.pth")
        torch.save(policy.state_dict(), path)
        print(f"policy trained in {time.perf_counter() - t0:.0f}s on {len(self.strat_memory):,} samples -> {path}")
        return policy

    def evaluate(self, policy):
        scores = evaluate_model(policy, self.device, self.args.eval_hands, seed=self.args.seed)
        for k, v in scores.items():
            self.log(f"eval/{k}", v, self.iteration)
        print("chips/hand vs", {k: round(v, 3) for k, v in scores.items()})
        with open(os.path.join(self.args.out, "eval.json"), "w") as f:
            json.dump({"iteration": self.iteration, "hands": self.args.eval_hands, "chips_per_hand": scores}, f, indent=2)
        return scores

    # -- main loop -------------------------------------------------------------------
    def run(self):
        a = self.args
        interrupted = False
        bar = tqdm(total=a.iterations, initial=self.iteration, desc="cfr")
        try:
            while self.iteration < a.iterations:
                self.cfr_iteration()
                s = self.last_stats
                post = dict(trav=f"{s['trav']:.1f}s", train=f"{s['train']:.1f}s", nodes=f"{s['nodes']:.0f}", strat=f"{len(self.strat_memory):,}")
                if a.eval_every and self.iteration % a.eval_every == 0:
                    scores = self.evaluate_iterate()
                    post["cur_vs_call"] = f"{scores['call']:+.2f}"
                if a.policy_eval_every and self.iteration % a.policy_eval_every == 0 and len(self.strat_memory) >= a.batch_size:
                    scores = self.evaluate_quick_policy()
                    post["avg_vs_call"] = f"{scores['call']:+.2f}"
                bar.set_postfix(**post)
                bar.update(1)
                if a.checkpoint_every and self.iteration % a.checkpoint_every == 0:
                    self.save_checkpoint()
        except KeyboardInterrupt:
            interrupted = True
            print("\ninterrupted - training the policy on the samples collected so far")
        finally:
            bar.close()
            self.runner.close()
        if a.checkpoint_every:
            self.save_checkpoint()
        elif interrupted:
            print("(no checkpoint written: pass --checkpoint-every N to keep resumable state)")
        if len(self.strat_memory) == 0:
            print("no strategy samples collected; nothing to train")
            return None
        policy = self.train_policy()
        if a.eval_hands > 0:
            self.evaluate(policy)
        if self.writer is not None:
            self.writer.close()
        return policy


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/deepcfr", help="output directory (policy.pth, checkpoint.pt, tensorboard)")
    p.add_argument("--iterations", type=int, default=300, help="CFR iterations")
    p.add_argument("--traversals", type=int, default=10_000, help="traversals per seat per iteration")
    p.add_argument("--workers", type=int, default=None, help="traversal workers (default: cores-1)")
    p.add_argument("--backend", default="auto", choices=["auto", "cpp", "python"], help="traversal backend")
    p.add_argument("--device", default=None, help="torch device: mps | cuda | cpu (default: auto)")
    p.add_argument("--adv-capacity", type=int, default=10_000_000, help="advantage memory size per seat")
    p.add_argument("--strat-capacity", type=int, default=10_000_000, help="strategy memory size")
    p.add_argument("--value-steps", type=int, default=4000, help="SGD steps per advantage-net fit")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--policy-epochs", type=int, default=50, help="epochs over the strategy memory for the policy net")
    p.add_argument("--eval-hands", type=int, default=100_000, help="hands per opponent for the final evaluation (0 = skip)")
    p.add_argument("--eval-every", type=int, default=5, help="evaluate the current strategy vs simple bots every N iterations (0 = off)")
    p.add_argument("--policy-eval-every", type=int, default=25, help="fit a quick average-strategy net and evaluate it every N iterations (0 = off)")
    p.add_argument("--policy-eval-epochs", type=int, default=2, help="epochs for the quick average-strategy fits")
    p.add_argument("--iterate-eval-hands", type=int, default=20_000, help="hands per opponent for the periodic evaluations")
    p.add_argument("--checkpoint-every", type=int, default=0, help="save a resumable checkpoint (nets + memories, up to ~150 bytes/sample) every N iterations (0 = off)")
    p.add_argument("--resume", default=None, help="checkpoint.pt to resume from")
    p.add_argument("--policy-only", default=None, help="skip CFR: train the policy from this checkpoint's strategy memory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile for the network fits")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    trainer = DeepCFRTrainer(args)
    if args.policy_only:
        trainer.load_checkpoint(args.policy_only)
        trainer.runner.close()
        policy = trainer.train_policy()
        if args.eval_hands > 0:
            trainer.evaluate(policy)
        return
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
