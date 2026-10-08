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
from headsup.deepcfr.memory import CircularBuffer, ReservoirBuffer
from headsup.deepcfr.traverse import TraversalRunner
from headsup.device import get_device, synchronize
from headsup.game import GameConfig, parse_bet_sizes
from headsup.model import ARCHS, CARDS, FEATURES, RM_FALLBACKS, BaseModel, count_parameters, normalize_config

# Hyperparameter presets (--preset); explicit flags always win.  "paper" = Brown et al. (2019)
# for FHP/HULH and the SD-CFR paper's average-strategy fit (20,000 updates x batch 20,480); "escher" = Table 3
# of McAleer et al. (2023): 1,000 regret and value trajectories, batch 2,048, 5,000 / 5,000 / 10,000 steps.
PRESETS = {
    "default": dict(traversals=10_000, batch_size=16384, value_steps=4000, adv_capacity=10_000_000,
                    strat_capacity=10_000_000, policy_epochs=50, policy_steps=None, policy_batch_size=None,
                    q_steps=1000, q_batch=512, value_trajectories=None),
    "paper": dict(traversals=10_000, batch_size=10_000, value_steps=4000, adv_capacity=40_000_000,
                  strat_capacity=40_000_000, policy_epochs=None, policy_steps=20_000, policy_batch_size=20_480,
                  q_steps=1000, q_batch=512, value_trajectories=None),
    "escher": dict(traversals=1000, batch_size=2048, value_steps=5000, adv_capacity=10_000_000,
                   strat_capacity=10_000_000, policy_epochs=None, policy_steps=10_000, policy_batch_size=2048,
                   q_steps=5000, q_batch=2048, value_trajectories=1000),
}


# ----------------------------------------------------------------------------- networks
def maybe_compile(model, enabled=True):
    """torch.compile roughly halves the step time on MPS; fall back to eager if it fails.

    On CUDA the fits are launch-bound (64-wide MLP, ~4 ms/step in eager), so the model is
    additionally captured into CUDA graphs (``mode="reduce-overhead"``: 4.5 -> 2.7 ms/step on an
    RTX 3090; identical numerics).  Every fit compiles a fresh model, which only re-records the
    graph (~0.5 s) - the inductor cache is shared.
    """
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        device = next(model.parameters()).device
        compiled = torch.compile(model, mode="reduce-overhead" if device.type == "cuda" else None)
        return compiled
    except Exception as exc:  # pragma: no cover
        print(f"torch.compile unavailable ({type(exc).__name__}: {exc}); using eager mode")
        return model


def _run_step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal=None):
    pred = fwd(obs)
    loss = loss_fn(pred, t, target) if legal is None else loss_fn(pred, t, target, legal)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=False)  # the foreach path is ~1000x slower on CPU in torch 2.13
    opt.step()
    return loss


def loss_weight_scale(iteration, power=1.0, mode="paper"):
    """Factor on the sample weights t^power of a fit made at iteration T.

    ``paper``: (power + 1) / T^power - for linear weights the paper's "we rescale all the batch weights by 2/T"
    (5.3).  The weights then average ~1 over the iterations 1..T, the loss stays O(1) and the gradient clip acts on
    outliers; with the ``raw`` weights (factor 1) the gradient norm grows with T and the clip at 1 rescales every
    step (measured on FHP at T = 450: norms 25-460 against 0.1-1.8)."""
    if mode == "raw" or iteration < 1:
        return 1.0
    return (power + 1.0) / float(iteration) ** power


def _weighted_mse(pred, t, target):
    return (t[:, None] * (pred - target).pow(2)).mean()


def _power_weighted_mse(power, masked=False, scale=1.0):
    """Sample weight ``scale`` t^power (power 1 = linear CFR; DCFR uses alpha = 1.5 for regrets).  ``masked``: the
    loss takes a 4th argument, the legal masks, and covers the legal actions only."""
    if masked:
        return lambda pred, t, target, legal: (scale * t.pow(power)[:, None] * legal * (pred - target).pow(2)).mean()
    if power == 1.0 and scale == 1.0:
        return _weighted_mse
    return lambda pred, t, target: (scale * t.pow(power)[:, None] * (pred - target).pow(2)).mean()


def train_advantage_net(
    buffer, device, steps, batch_size, lr=1e-3, grad_clip=1.0, log=None, tag="", compile=True, log_step_offset=0, log_every=100,
    model_config=None, weight_power=1.0, target_scale="auto", masked=False, iteration=0, loss_weights="paper", zero_head=False,
):
    """Fresh network fitted to (obs -> regrets) with iteration-weighted MSE (weight t^weight_power, rescaled for
    the fit of ``iteration`` as ``loss_weights`` says: :func:`loss_weight_scale`).  The network starts "from a
    random initialization" (paper 5.2) unless ``zero_head``.

    ``target_scale``: the regrets are divided by it for the fit and the output layer is multiplied
    by it afterwards, so the network still predicts chips.  From a zero-initialised head, Adam's
    ~lr-sized steps cannot reach outputs of hundreds of chips within a few thousand steps (FHP's
    regrets have an RMS of ~270 chips: the raw fit explained half the variance the scaled one
    does); regret matching is scale-free, so only the fit quality changes.  ``"auto"`` = the RMS of
    the memory's targets, ``None`` / 1 = raw chips.

    ``masked``: the loss covers each sample's legal actions only (the buffer's legal masks).  Otherwise the
    outputs of illegal actions are fitted to their twins' regrets (fold = check when nothing is to be called),
    which the network never uses; the SD-CFR / DREAM / ESCHER authors' networks multiply them by 0.

    Returns ``(model, final_loss)`` (the loss in scaled units).  Per-step losses are logged at
    ``log_step_offset + step`` so successive fits form one continuous curve in TensorBoard.
    """
    if target_scale == "auto":
        target_scale = 1.0
        if len(buffer):
            _, _, sample, *legal = buffer.sample(min(65536, len(buffer)), with_legal=masked)
            sq = sample.pow(2) * legal[0] if masked else sample.pow(2)
            target_scale = max(float((sq.sum() / (legal[0].sum() if masked else sq.numel())).sqrt()), 1e-6)
    scale = float(target_scale or 1.0)
    if len(buffer) == 0:  # no samples yet (e.g. the opponent folds every hand before this seat acts): uniform net
        return BaseModel(config=model_config).to(device).eval(), float("nan")
    model = BaseModel(config=model_config, zero_head=zero_head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    fwd = maybe_compile(model, compile)
    loss_fn = _power_weighted_mse(weight_power, masked, loss_weight_scale(iteration, weight_power, loss_weights))
    loss = None
    for step, (obs, t, target, *legal) in enumerate(buffer.prefetch(batch_size, steps, with_legal=masked)):
        legal = legal[0] if legal else None
        if scale != 1.0:
            target = target / scale
        try:
            loss = _run_step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal)
        except Exception as exc:
            if fwd is model:
                raise
            print(f"compiled step failed ({type(exc).__name__}); falling back to eager")
            fwd = model
            loss = _run_step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal)
        if log is not None and step % log_every == 0:
            log(f"advantage{tag}/loss", loss.item(), log_step_offset + step)
    model.eval()
    if scale != 1.0:
        with torch.no_grad():
            model.action_head.weight.mul_(scale)
            model.action_head.bias.mul_(scale)
    final_loss = float(loss.item()) if loss is not None else float("nan")
    if log is not None and steps:
        log(f"advantage{tag}/loss", final_loss, log_step_offset + steps - 1)
        log(f"advantage{tag}/target_scale", scale, log_step_offset + steps - 1)
    return model, final_loss


def _policy_loss(power=1.0, masked=False, scale=1.0):
    """Weighted squared error between softmax(logits) and the stored strategy (weight ``scale`` t^power).  ``masked``:
    a 4th argument, the legal masks - the softmax runs over the legal actions only, as the strategy is used."""
    def weights(t):
        return scale * (t if power == 1.0 else t.pow(power))

    if masked:
        def loss(logits, t, target, legal):
            probs = torch.softmax(logits.masked_fill(legal == 0, -1e20), dim=-1)
            return (weights(t)[:, None] * (probs - target).pow(2)).mean()
    else:
        def loss(logits, t, target):
            return (weights(t)[:, None] * (torch.softmax(logits, dim=-1) - target).pow(2)).mean()

    return loss


def train_policy_net(buffer, device, epochs, batch_size, lr=1e-3, lr_decay=None, log=None, progress=True, compile=True,
                     model_config=None, steps=None, weight_power=1.0, grad_clip=1.0, masked=False, iteration=0, loss_weights="paper"):
    """Average-strategy network: softmax(logits) fitted to stored strategies (weighted MSE, weight
    t^weight_power: 1 = linear CFR average, 2 = DCFR's quadratic strategy weighting; rescaled as the advantage
    fits are, :func:`loss_weight_scale`).  ``masked``: softmax over each sample's legal actions (the buffer's masks).

    ``epochs`` passes over the memory, or a fixed number of ``steps``.  The learning rate is constant and the
    gradient norm clipped, as in the authors' code ("no lr decay"); ``lr_decay`` = the factor per epoch (per 2 %
    of the steps) of a step schedule (0.9 was the default here before: it leaves 20 % of the lr integral).
    """
    model = BaseModel(config=model_config, zero_head=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if steps is None:
        steps_per_epoch = max(1, len(buffer) // batch_size)
        steps = epochs * steps_per_epoch
    else:
        steps_per_epoch = max(1, steps // 50)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=steps_per_epoch, gamma=lr_decay) if lr_decay else None
    model.train()
    fwd = maybe_compile(model, compile)
    loss_fn = _policy_loss(weight_power, masked, loss_weight_scale(iteration, weight_power, loss_weights))
    it = tqdm(range(steps), desc="policy", disable=not progress, leave=False)
    batches = buffer.prefetch(batch_size, steps, with_legal=masked)
    for step in it:
        obs, t, target, *legal = next(batches)
        legal = legal[0] if legal else None
        try:
            loss = _run_step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal)
        except Exception as exc:
            if fwd is model:
                raise
            print(f"compiled step failed ({type(exc).__name__}); falling back to eager")
            fwd = model
            loss = _run_step(fwd, model, opt, obs, t, target, loss_fn, grad_clip, legal)
        if scheduler is not None:
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

        self.algo = args.algo
        # DeepCFR / SD-CFR: external sampling.  DREAM: outcome sampling with learned baselines,
        # SD-CFR averaging (the paper's setting).  ESCHER: outcome sampling with a history value net,
        # DeepCFR-style average policy net (+ the iterate bank for evaluation).
        self.use_deepcfr = self.algo in ("deepcfr", "both", "escher")
        self.use_sdcfr = self.algo in ("sdcfr", "both", "dream", "escher")
        if getattr(args, "game_config", None):  # a resumed run: the checkpoint's own game, whatever the presets are now
            self.game = GameConfig.from_dict(args.game_config)
        elif args.game and args.game not in ("nlhe", "holdem"):  # limit presets (FHP / HULH)
            from headsup.games.holdem import make_holdem

            self.game = make_holdem(args.game)
        else:
            bet_sizes = parse_bet_sizes(args.bet_sizes)
            mask = args.mask_redundant == "on" or (args.mask_redundant == "auto" and bet_sizes != ("min",))
            self.game = GameConfig(raise_cap=args.raise_cap, bet_sizes=bet_sizes, mask_redundant=mask)
        self.model_config = normalize_config(
            dict(features=args.features, arch=args.net, cards=args.cards, dim=args.dim, rm_fallback=args.rm_fallback, game=self.game)
        )
        self.nets = [BaseModel(config=self.model_config).to(self.device).eval() for _ in range(2)]
        obs_dim, num_actions = self.nets[0].obs_dim, self.nets[0].num_actions
        mem_dev = torch.device(args.memory_device) if args.memory_device else self.device
        self.masked_loss = bool(getattr(args, "masked_loss", False))
        self.adv_memory = [
            ReservoirBuffer(args.adv_capacity, mem_dev, obs_dim=obs_dim, target_dim=num_actions, seed=args.seed + i, sample_device=self.device,
                            legal_dim=num_actions if self.masked_loss else 0)
            for i in range(2)
        ]
        self.strat_memory = (
            ReservoirBuffer(args.strat_capacity, mem_dev, obs_dim=obs_dim, target_dim=num_actions, seed=args.seed + 2, sample_device=self.device,
                            legal_dim=num_actions if self.masked_loss else 0)
            if self.use_deepcfr else None
        )
        # SD-CFR: keep every iteration's advantage net.  The bank starts with the untrained net (iteration 0: it plays
        # iteration 1 and has weight 0 in the average); ``iterate_first`` = the iteration of the bank's first net
        self.iterates = [[n.state_dict_cpu()] for n in self.nets] if self.use_sdcfr else None
        self.iterate_first = 0
        self.runner = TraversalRunner(args.workers, backend=args.backend, game=self.game)
        # history value networks (both players' cards): DREAM baselines Q_p(h, a) per player, the
        # ESCHER value net q(h, a) (player 0's return; re-fitted every iteration on fresh trajectories)
        self.value_config = normalize_config(dict(self.model_config, opp_cards=True))
        self.value_nets, self.value_opts, self.value_memory = [], [], []
        if self.algo in ("dream", "escher"):
            n_value = 2 if self.algo == "dream" else 1
            self.value_nets = [BaseModel(config=self.value_config).to(self.device).eval() for _ in range(n_value)]
            self.value_opts = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in self.value_nets]
            self.value_memory = [CircularBuffer(args.q_capacity, self.device, self.value_nets[0].obs_dim, seed=args.seed + 10 + i)
                                 for i in range(n_value)]
            if self.runner.backend != "cpp":
                raise SystemExit("DREAM / ESCHER need the C++ extension (python setup.py build_ext)")
        self.value_scales = [None] * len(self.value_nets)
        cfg = self.model_config
        print(
            f"device={self.device}  algo={self.algo}  traversal backend={self.runner.backend} x{self.runner.num_workers} workers  "
            f"memories: adv {args.adv_capacity:,} x2" + (f", strat {args.strat_capacity:,}" if self.use_deepcfr else "")
            + f" on {mem_dev}"
        )
        print(
            f"network: features={cfg['features']} arch={cfg['arch']} cards={cfg['cards']} dim={cfg['dim']} "
            f"rm_fallback={cfg['rm_fallback']}  ({count_parameters(self.nets[0]):,} parameters, obs[{obs_dim}])"
        )
        print(f"game: {args.game} bet sizes {self.game.bet_sizes} ({num_actions} actions), raise cap(s) "
              f"{self.game.raise_caps or self.game.raise_cap}, rounds {self.game.num_rounds}, all-in {self.game.all_in}, "
              f"mask redundant raises {self.game.mask_redundant}")

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
            "strat_memory": self.strat_memory.state_dict() if self.strat_memory is not None else None,
            "iterates": self._stacked_iterates() if self.iterates is not None else None,
            "iterate_first": self.iterate_first,
            "args": vars(self.args),
            "model_config": dict(self.model_config),
            "value_nets": [n.state_dict() for n in self.value_nets],
            "value_opts": [o.state_dict() for o in self.value_opts],
            "value_scales": list(self.value_scales),
            "value_memory": [m.state_dict() for m in self.value_memory],
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        if self.iterates is not None:
            self.save_iterates()
        return path

    def _iterate_ids(self):
        """The iteration each net of the bank was trained in."""
        return [self.iterate_first + i for i in range(len(self.iterates[0]))]

    def _bank(self):
        from headsup.sdcfr import IterateBank

        return IterateBank(self._stacked_iterates(), self.device, self.model_config, iterations=self._iterate_ids())

    def _stacked_iterates(self):
        return {s: {k: torch.stack([sd[k] for sd in dicts]) for k in dicts[0]} for s, dicts in enumerate(self.iterates)}

    def save_iterates(self, path=None):
        """Write the SD-CFR iterate bank (all advantage nets) in IterateBank format."""
        path = path or os.path.join(self.args.out, "iterates.pt")
        stacked = self._stacked_iterates()
        torch.save({"seats": stacked, "T": len(self.iterates[0]), "config": dict(self.model_config), "iterations": self._iterate_ids()}, path)
        return path

    def sdcfr_player(self, mode="sample", seed=None):
        from headsup.sdcfr import SDCFRPlayer

        bank = self._bank()
        return SDCFRPlayer(bank, mode=mode, seed=seed)

    @staticmethod
    def checkpoint_model_config(path):
        """Model variant a checkpoint was trained with."""
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return normalize_config(state["model_config"])

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if normalize_config(state["model_config"]) != self.model_config:
            raise ValueError(f"checkpoint was trained with {state['model_config']}, this run uses {self.model_config}")
        self.iteration = int(state["iteration"])
        for n, sd in zip(self.nets, state["nets"]):
            n.load_state_dict(sd)
        for m, sd in zip(self.adv_memory, state["adv_memory"]):
            m.load_state_dict(sd)
        if self.strat_memory is not None:
            if state.get("strat_memory") is not None:
                self.strat_memory.load_state_dict(state["strat_memory"])
            else:
                print("(checkpoint has no strategy memory: the policy net will only see iterations from here on)")
        # a checkpoint of another algorithm may hold fewer value nets (none): the others start fresh
        loaded = min(len(self.value_nets), len(state.get("value_nets", [])))
        for n, sd in zip(self.value_nets, state.get("value_nets", [])):
            n.load_state_dict(sd)
        for o, sd in zip(self.value_opts, state.get("value_opts", [])):
            o.load_state_dict(sd)
            for group in o.param_groups:  # the optimiser state keeps its moments, the run's --lr applies
                group["lr"] = self.args.lr
        for m, sd in zip(self.value_memory, state.get("value_memory", [])):
            m.load_state_dict(sd)
        # value nets trained before the scaled fits were introduced predict raw chips (scale 1)
        scales = list(state.get("value_scales", [1.0] * loaded))
        self.value_scales = [scales[i] if i < min(loaded, len(scales)) else None for i in range(len(self.value_nets))]
        # a fresh stream of traversal seeds (the sequence would restart with the first iterations' seeds)
        self.seed_seq = np.random.SeedSequence([int(self.args.seed), self.iteration])
        if self.iterates is not None:
            stacked = state.get("iterates")
            if stacked is not None:
                self.iterates = [
                    [{k: v[t].clone() for k, v in d.items()} for t in range(next(iter(d.values())).shape[0])]
                    for _, d in sorted(stacked.items())
                ]
                self.iterate_first = int(state.get("iterate_first", 0))
            else:  # the bank starts with the checkpoint's nets, as the nets of its iteration
                self.iterates = [[n.state_dict_cpu()] for n in self.nets]
                self.iterate_first = self.iteration
                print("(checkpoint has no iterate bank: SD-CFR average will only cover iterations from here on)")
        print(f"resumed from {path} at iteration {self.iteration}")

    # -- one CFR iteration -----------------------------------------------------------
    def _seed(self):
        return int(self.seed_seq.spawn(1)[0].generate_state(1)[0])

    def train_value_net(self, i, steps=None, batch=None):
        """Continue training value net ``i`` on its FIFO (masked MSE on the taken action's output;
        DREAM / ESCHER: Adam 1e-3, grad clip 1, 1000 x 512 per iteration by default).  The fit runs in
        units of a per-net scale (the RMS of the first targets it sees; see train_advantage_net: a
        zero-initialised head cannot reach FHP's hundreds of chips) - the output layer is divided by
        it for the fit and multiplied back afterwards, so the net always predicts chips."""
        a = self.args
        steps, batch = steps or a.q_steps, batch or a.q_batch
        net, opt, mem = self.value_nets[i], self.value_opts[i], self.value_memory[i]
        if len(mem) < batch:
            return float("nan")
        if self.value_scales[i] is None:
            _, _, sample = mem.sample(min(65536, len(mem)))
            self.value_scales[i] = max(float(sample.pow(2).mean().sqrt()), 1e-6) if a.target_scale != "none" else 1.0
        scale = self.value_scales[i]
        with torch.no_grad():
            net.action_head.weight.div_(scale)
            net.action_head.bias.div_(scale)
        net.train()
        loss_val = float("nan")
        for _ in range(steps):
            obs, action, target = mem.sample(batch)
            pred = net(obs).gather(1, action[:, None]).squeeze(1)
            loss = torch.mean((pred - target / scale) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, foreach=False)
            opt.step()
            loss_val = float(loss.detach())
        net.eval()
        with torch.no_grad():
            net.action_head.weight.mul_(scale)
            net.action_head.bias.mul_(scale)
        return loss_val

    def _sample_seat(self, seat, weights, t):
        """Collect this iteration's samples for ``seat`` with the algorithm's sampler; returns
        (adv Samples, strat Samples or None, nodes)."""
        a = self.args
        if self.algo == "dream":
            adv, val, nodes = self.runner.collect_dream(weights, self.value_nets[seat].numpy_weights(), seat, a.traversals, t, a.epsilon, self._seed())
            self.value_memory[seat].add(val.obs, val.t, val.target[np.arange(len(val)), val.t.astype(int)])
            self.log(f"value/seat{seat}/loss", self.train_value_net(seat), self.iteration)
            return adv, None, nodes
        if self.algo == "escher":
            adv, strat, _, nodes = self.runner.collect_escher_regrets(weights, self.value_nets[0].numpy_weights(), seat, a.traversals, t, self._seed())
            return adv, strat, nodes
        adv, strat, nodes = self.runner.collect(weights, seat, a.traversals, t, self._seed())
        return adv, strat, nodes

    def cfr_iteration(self):
        a = self.args
        # what an interrupt inside this iteration has to restore (see _rollback)
        self._undo = (self.iteration, list(self.nets), [len(it) for it in self.iterates] if self.iterates is not None else None)
        self.iteration += 1
        t = float(self.iteration)  # linear CFR weight
        weights = [n.numpy_weights() for n in self.nets]
        if self.algo == "escher":  # a fresh value net on this iteration's trajectories (ESCHER's reference code)
            t0 = time.perf_counter()
            val, _ = self.runner.collect_escher_values(weights, a.value_trajectories or a.traversals, self._seed(), a.value_epsilon)
            if len(val) > self.value_memory[0].capacity:  # this iteration's trajectories, all of them (--q-capacity is the
                self.value_memory[0] = CircularBuffer(len(val), self.device, self.value_nets[0].obs_dim, seed=a.seed + 10)  # DREAM FIFO's)
            self.value_memory[0].clear()
            self.value_memory[0].add(val.obs, val.t, val.target[np.arange(len(val)), val.t.astype(int)])
            self.value_nets[0] = BaseModel(config=self.value_config).to(self.device).eval()
            self.value_opts[0] = torch.optim.Adam(self.value_nets[0].parameters(), lr=a.lr)
            self.value_scales[0] = None
            self.log("value/loss", self.train_value_net(0), self.iteration)
            self.log("time/value", time.perf_counter() - t0, self.iteration)
        for seat in range(2):
            t0 = time.perf_counter()
            adv, strat, nodes = self._sample_seat(seat, weights, t)
            t_trav = time.perf_counter() - t0
            if self.masked_loss and len(adv) and adv.legal is None:
                raise RuntimeError("--masked-loss: the traversal returned no legal masks")
            self.adv_memory[seat].add(adv.obs, adv.t, adv.target, adv.legal if self.masked_loss else None)
            if self.strat_memory is not None and strat is not None:
                if self.masked_loss and len(strat) and strat.legal is None:
                    raise RuntimeError("--masked-loss: the traversal returned no legal masks for the strategy samples")
                self.strat_memory.add(strat.obs, strat.t, strat.target, strat.legal if self.masked_loss else None)

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
                model_config=self.model_config,
                weight_power=a.regret_power,
                iteration=self.iteration,
                loss_weights=a.loss_weights,
                target_scale=None if a.target_scale == "none" else ("auto" if a.target_scale == "auto" else float(a.target_scale)),
                masked=self.masked_loss,
            )
            weights[seat] = self.nets[seat].numpy_weights()
            if self.iterates is not None:
                self.iterates[seat].append(self.nets[seat].state_dict_cpu())
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
            self.log(f"samples/strat_per_traversal/seat{seat}", (len(strat) if strat is not None else 0) / a.traversals, it)
            self.log(f"samples/nodes_per_traversal/seat{seat}", nodes / a.traversals, it)
            self.log(f"memory/adv/seat{seat}", len(self.adv_memory[seat]), it)
            if self.strat_memory is not None:
                self.log("memory/strat", len(self.strat_memory), it)
            self.last_stats = dict(trav=t_trav, train=t_train, nodes=nodes / a.traversals, adv=len(adv),
                                   strat=len(strat) if strat is not None else 0)
        self._undo = None

    def _rollback(self):
        """After an interrupt inside an iteration: back to the last complete one - the counter, the current nets and
        the iterate bank (seat 0 may already have its new net).  The interrupted iteration's samples stay in the
        memories (reservoir replacement cannot be undone); a resumed run repeats that iteration."""
        if getattr(self, "_undo", None) is None:
            return
        self.iteration, self.nets, lengths = self._undo
        if lengths is not None:
            for bank, n in zip(self.iterates, lengths):
                del bank[n:]
        self._undo = None

    def _advantage_fit_quality(self, seat, n=65536):
        """Unweighted MSE of the freshly fitted net on a memory sample, and the target scale (over the fitted
        outputs: the legal actions with --masked-loss)."""
        if len(self.adv_memory[seat]) == 0:
            return float("nan"), float("nan")
        obs, _, target, *legal = self.adv_memory[seat].sample(min(n, len(self.adv_memory[seat])), with_legal=self.masked_loss)
        m = legal[0] if legal else torch.ones_like(target)
        with torch.no_grad():
            pred = self.nets[seat](obs)
            mse = ((pred - target).pow(2) * m).sum().item() / m.sum().item()
            rms = (target.pow(2) * m).sum().div(m.sum()).sqrt().item()
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

    def evaluate_sdcfr(self):
        """SD-CFR average strategy (trajectory sampling over the iterate bank) vs simple bots."""
        from headsup.deepcfr.evaluate import evaluate

        player = self.sdcfr_player(mode="sample", seed=self.args.seed + self.iteration)
        scores = evaluate(player, self.args.iterate_eval_hands, num_envs=1024, seed=self.args.seed)
        for k, v in scores.items():
            self.log(f"eval_sdcfr/{k}", v, self.iteration)
        return scores

    def evaluate_lbr(self, hands, policy=None, tag="lbr", progress=False):
        """Local best response (headsup.lbr) vs the current strategy, the SD-CFR average and
        optionally a policy net: LBR's chips/hand = an exploitability lower bound (lower is better)."""
        from headsup.lbr import LocalBestResponse
        from headsup.players import RegretMatchingPlayer, TorchPolicyPlayer

        a = self.args
        targets = {}
        cur = RegretMatchingPlayer(self.nets, device=self.device, seed=a.seed)
        targets["current_strategy"] = (cur, cur)
        if self.iterates is not None:
            from headsup.sdcfr import SDCFRPlayer

            bank = self._bank()
            targets["sdcfr"] = (SDCFRPlayer(bank, mode="sample", seed=a.seed), SDCFRPlayer(bank.thin(a.lbr_model_iterates), mode="exact"))
        if policy is not None:
            pol = TorchPolicyPlayer(policy, device=self.device, seed=a.seed)
            targets["avg_strategy"] = (pol, pol)
        out = {}
        for name, (opponent, model) in targets.items():
            lbr = LocalBestResponse(name, num_tables=a.lbr_tables, device=self.device, seed=a.seed + self.iteration,
                                    workers=16, opponent=opponent, model=model)
            r = lbr.play(hands, progress=progress)
            out[name] = (float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))))
            self.log(f"{tag}/{name}", out[name][0], self.iteration)
        return out

    def evaluate_quick_policy(self):
        """Quick average-strategy fit (few epochs) vs simple bots: the signal CFR actually improves."""
        a = self.args
        policy = train_policy_net(
            self.strat_memory, self.device, a.policy_eval_epochs, a.batch_size, a.lr, compile=not a.no_compile, progress=False,
            model_config=self.model_config, weight_power=a.strategy_power, **self._policy_fit_options(),
        )
        scores = evaluate_model(policy, self.device, a.iterate_eval_hands, seed=a.seed)
        for k, v in scores.items():
            self.log(f"eval_avg_strategy/{k}", v, self.iteration)
        return scores

    # -- policy ----------------------------------------------------------------------
    def _policy_fit_options(self):
        a = self.args
        return dict(lr_decay=a.policy_lr_decay, masked=self.masked_loss and self.strat_memory.legal is not None,
                    iteration=self.iteration, loss_weights=a.loss_weights)

    def train_policy(self):
        a = self.args
        t0 = time.perf_counter()
        policy = train_policy_net(
            self.strat_memory, self.device, a.policy_epochs, a.policy_batch_size or a.batch_size, a.lr, log=self.log,
            compile=not a.no_compile, model_config=self.model_config, steps=a.policy_steps, weight_power=a.strategy_power,
            **self._policy_fit_options(),
        )
        synchronize(self.device)
        path = os.path.join(a.out, "policy.pth")
        policy.save(path)
        print(f"policy trained in {time.perf_counter() - t0:.0f}s on {len(self.strat_memory):,} samples -> {path}")
        return policy

    def evaluate(self, policy):
        scores = evaluate_model(policy, self.device, self.args.eval_hands, seed=self.args.seed)
        for k, v in scores.items():
            self.log(f"eval/{k}", v, self.iteration)
        print("DeepCFR policy net, chips/hand vs", {k: round(v, 3) for k, v in scores.items()})
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
                post = dict(trav=f"{s['trav']:.1f}s", train=f"{s['train']:.1f}s", nodes=f"{s['nodes']:.0f}",
                            strat=f"{len(self.strat_memory):,}" if self.strat_memory is not None else "-")
                if a.eval_every and self.iteration % a.eval_every == 0:
                    scores = self.evaluate_iterate()
                    post["cur_vs_call"] = f"{scores['call']:+.2f}"
                if a.policy_eval_every and self.iteration % a.policy_eval_every == 0:
                    if self.strat_memory is not None and len(self.strat_memory) >= a.batch_size:
                        scores = self.evaluate_quick_policy()
                        post["avg_vs_call"] = f"{scores['call']:+.2f}"
                    if self.iterates is not None:
                        scores = self.evaluate_sdcfr()
                        post["sdcfr_vs_call"] = f"{scores['call']:+.2f}"
                if a.lbr_every and self.iteration % a.lbr_every == 0:
                    lbr = self.evaluate_lbr(a.lbr_hands)
                    post["lbr"] = "/".join(f"{v[0]:+.2f}" for v in lbr.values())
                bar.set_postfix(**post)
                bar.update(1)
                if a.checkpoint_every and self.iteration % a.checkpoint_every == 0:
                    self.save_checkpoint()
        except KeyboardInterrupt:
            interrupted = True
            self._rollback()
            print(f"\ninterrupted - back at iteration {self.iteration}; training the policy on the samples collected so far")
        finally:
            bar.close()
            self.runner.close()
        if a.checkpoint_every:
            self.save_checkpoint()
        elif interrupted:
            print("(no checkpoint written: pass --checkpoint-every N to keep resumable state)")
        return self.finish()

    def finish(self):
        """Produce the final artefacts: DeepCFR policy net and/or SD-CFR iterate bank, then evaluate."""
        a = self.args
        results = {"iteration": self.iteration, "hands": a.eval_hands, "algo": self.algo, "model_config": dict(self.model_config)}
        policy = None
        if self.iterates is not None:
            path = self.save_iterates()
            print(f"SD-CFR iterate bank: {len(self.iterates[0])} nets per seat -> {path}")
        if self.strat_memory is not None:
            if len(self.strat_memory) == 0:
                print("no strategy samples collected; skipping the policy net")
            else:
                policy = self.train_policy()
        if a.eval_hands > 0:
            from headsup.env import make_vec_env, play_hands
            from headsup.players import TorchPolicyPlayer

            if policy is not None:
                results["deepcfr_chips_per_hand"] = self.evaluate(policy)
            if self.iterates is not None:
                sd = self.evaluate_sdcfr_final()
                results["sdcfr_chips_per_hand"] = sd
            if a.lbr_final_hands > 0:
                lbr = self.evaluate_lbr(a.lbr_final_hands, policy=policy, tag="lbr_final", progress=True)
                results["lbr_chips_per_hand"] = {k: v[0] for k, v in lbr.items()}
                results["lbr_se"] = {k: v[1] for k, v in lbr.items()}
                print("LBR (exploitability lower bound), chips/hand:", {k: f"{v[0]:+.3f} ± {v[1]:.3f}" for k, v in lbr.items()})
            if policy is not None and self.iterates is not None:
                sd_player = self.sdcfr_player(mode="sample", seed=a.seed + 1)
                r = play_hands(make_vec_env(1024, sd_player, seed=a.seed, game=self.game), TorchPolicyPlayer(policy, device=self.device, seed=a.seed), a.eval_hands)
                h2h = float(r.mean())
                se = float(r.std() / np.sqrt(len(r)))
                results["deepcfr_vs_sdcfr_chips_per_hand"] = h2h
                results["deepcfr_vs_sdcfr_se"] = se
                self.log("eval/deepcfr_vs_sdcfr", h2h, self.iteration)
                print(f"head-to-head DeepCFR policy vs SD-CFR average: {h2h:+.3f} ± {se:.3f} chips/hand")
            with open(os.path.join(a.out, "eval.json"), "w") as f:
                json.dump(results, f, indent=2)
        if self.writer is not None:
            self.writer.close()
        return policy

    def evaluate_sdcfr_final(self):
        from headsup.deepcfr.evaluate import evaluate

        player = self.sdcfr_player(mode="sample", seed=self.args.seed)
        scores = evaluate(player, self.args.eval_hands, seed=self.args.seed)
        for k, v in scores.items():
            self.log(f"eval_sdcfr/{k}", v, self.iteration)
        print("SD-CFR average strategy, chips/hand vs", {k: round(v, 3) for k, v in scores.items()})
        return scores


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/deepcfr", help="output directory (policy.pth, checkpoint.pt, tensorboard)")
    p.add_argument("--algo", default="both", choices=["deepcfr", "sdcfr", "both", "dream", "escher"],
                   help="deepcfr: strategy memory + policy net; sdcfr: keep all iterates (Single Deep CFR); both (default); "
                        "dream: outcome sampling with learned baselines + SD-CFR averaging (Steinberger et al. 2020); "
                        "escher: outcome sampling with a history value net + average policy net (McAleer et al. 2023)")
    p.add_argument("--epsilon", type=float, default=0.5, help="DREAM: exploration of the traverser (xi = eps * uniform + (1 - eps) * sigma)")
    p.add_argument("--q-steps", type=int, default=None, help="DREAM / ESCHER: value-net SGD steps per iteration (default 1000, the DREAM paper's)")
    p.add_argument("--q-batch", type=int, default=None, help="DREAM / ESCHER: value-net batch size (default 512, the DREAM paper's)")
    p.add_argument("--q-capacity", type=int, default=200_000, help="DREAM / ESCHER: value-net FIFO capacity (paper: 200 000)")
    p.add_argument("--value-trajectories", type=int, default=None, help="ESCHER: value trajectories per iteration (default: --traversals)")
    p.add_argument("--value-epsilon", type=float, default=0.01, help="ESCHER: uniform exploration of the value trajectories (reference code: 0.01)")
    p.add_argument("--iterations", type=int, default=300, help="CFR iterations")
    p.add_argument("--preset", default="default", choices=sorted(PRESETS),
                   help="hyperparameter preset; 'paper' = DeepCFR / SD-CFR papers (10k traversals, batch 10k, 40M memories, "
                        "policy 20k updates x 20480); 'escher' = the ESCHER paper's Table 3 (1,000 regret and value trajectories, "
                        "batch 2,048, 5,000 regret / 5,000 value / 10,000 policy steps); explicit flags override")
    p.add_argument("--traversals", type=int, default=None, help="traversals per seat per iteration (default 10,000)")
    p.add_argument("--workers", type=int, default=None, help="traversal workers (default: cores-1)")
    p.add_argument("--backend", default="auto", choices=["auto", "cpp", "python"], help="traversal backend")
    p.add_argument("--device", default=None, help="torch device: mps | cuda | cpu (default: auto)")
    p.add_argument("--memory-device", default=None,
                   help="where the reservoir memories live (default: the training device); 'cpu' for memories that do not "
                        "fit on the GPU (e.g. --preset paper with history features): batches are gathered in a background thread")
    p.add_argument("--adv-capacity", type=int, default=None, help="advantage memory size per seat (default 10M)")
    p.add_argument("--strat-capacity", type=int, default=None, help="strategy memory size (default 10M)")
    p.add_argument("--value-steps", type=int, default=None, help="SGD steps per advantage-net fit (default 4000)")
    p.add_argument("--batch-size", type=int, default=None, help="advantage-net batch size (default 16384)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-scale", default="auto",
                   help="advantage-net fits: divide the regrets by this (auto = their RMS) and scale the output layer back; 'none' = raw chips")
    p.add_argument("--masked-loss", action=argparse.BooleanOptionalAction, default=None,
                   help="losses on legal actions only: the advantage nets' illegal outputs are not fitted and the policy net's softmax "
                        "runs over the legal actions (the SD-CFR / DREAM / ESCHER authors' nets mask them; default: on for --algo "
                        "dream / escher, whose samplers store 0 for illegal actions, off otherwise)")
    p.add_argument("--loss-weights", default=None, choices=["paper", "raw"],
                   help="sample weights of the fits at iteration T: paper = t rescaled by 2/T (DeepCFR 5.3; default) | raw = t "
                        "(the loss and its gradient grow with T; the behaviour before 2026-10-08)")
    p.add_argument("--policy-lr-decay", type=float, default=None,
                   help="policy-net fit: learning-rate factor per epoch (per 2 %% of --policy-steps); default: constant lr, as the "
                        "authors (0.9 = the earlier schedule here)")
    p.add_argument("--regret-power", type=float, default=1.0,
                   help="advantage samples of iteration t weigh t^power in the fit (1 = linear CFR; DCFR-style alpha = 1.5)")
    p.add_argument("--strategy-power", type=float, default=1.0,
                   help="strategy samples weigh t^power in the policy-net fit (1 = linear average; 2 = DCFR's quadratic weighting)")
    p.add_argument("--policy-epochs", type=int, default=None, help="epochs over the strategy memory for the policy net (default 50)")
    p.add_argument("--policy-steps", type=int, default=None, help="fixed number of policy-net updates instead of epochs")
    p.add_argument("--policy-batch-size", type=int, default=None, help="policy-net batch size (default: --batch-size)")
    # network variant (see headsup.model.BaseModel)
    p.add_argument("--features", default="aggregated", choices=FEATURES,
                   help="bet features: aggregated (8 pot-normalised totals) | history (DeepCFR per-street bet history) | both")
    p.add_argument("--net", default=None, choices=ARCHS,
                   help="architecture: current (3 branches; default) | paper (Brown et al. Fig. 1 as Appendix C prints it) | deepcfr (the "
                        "network the paper trained: Appendix C with a 3x wider card branch, 98,948-parameter family; --preset paper)")
    p.add_argument("--cards", default="embed", choices=CARDS, help="card input: embed (rank+suit+card embeddings) | onehot (SD-CFR)")
    p.add_argument("--dim", type=int, default=64, help="hidden / embedding width")
    p.add_argument("--rm-fallback", default=None, choices=RM_FALLBACKS,
                   help="regret matching when no advantage is positive: uniform | argmax (DeepCFR / DREAM papers; the default with "
                        "--preset paper and for --algo dream / escher)")
    # game (action tree; stored in the model config)
    p.add_argument("--game", default="nlhe", choices=["nlhe", "fhp", "hulh"],
                   help="nlhe: the no-limit abstraction below; fhp / hulh: the DeepCFR paper's limit games (blinds 50/100)")
    p.add_argument("--bet-sizes", default="min",
                   help="raise sizes between check/call and all-in: 'min' (call + 1 BB, the original game) and/or pot "
                        "fractions, e.g. '0.5,1,2' or 'min,1' (at most 5)")
    p.add_argument("--raise-cap", type=int, default=3, help="the N-th consecutive raise becomes an all-in")
    p.add_argument("--mask-redundant", default="auto", choices=["auto", "on", "off"],
                   help="hide raises that duplicate another action (all-in / smaller size / capped); auto = on for "
                        "custom bet sizes, off for the original min-raise game")
    p.add_argument("--eval-hands", type=int, default=100_000, help="hands per opponent for the final evaluation (0 = skip)")
    p.add_argument("--eval-every", type=int, default=5, help="evaluate the current strategy vs simple bots every N iterations (0 = off)")
    p.add_argument("--policy-eval-every", type=int, default=25, help="fit a quick average-strategy net and evaluate it every N iterations (0 = off)")
    p.add_argument("--policy-eval-epochs", type=int, default=2, help="epochs for the quick average-strategy fits")
    p.add_argument("--iterate-eval-hands", type=int, default=20_000, help="hands per opponent for the periodic evaluations")
    p.add_argument("--lbr-every", type=int, default=0, help="run a local best response vs the current strategy / SD-CFR average every N iterations (0 = off)")
    p.add_argument("--lbr-hands", type=int, default=1000, help="duplicate hand pairs per periodic LBR evaluation")
    p.add_argument("--lbr-final-hands", type=int, default=5000, help="duplicate hand pairs for the final LBR evaluation (0 = skip)")
    p.add_argument("--lbr-tables", type=int, default=64)
    p.add_argument("--lbr-model-iterates", type=int, default=32, help="SD-CFR iterates LBR queries (thinned bank; the opponent plays all)")
    p.add_argument("--checkpoint-every", type=int, default=0, help="save a resumable checkpoint (nets + memories, up to ~75 bytes/sample) every N iterations (0 = off)")
    p.add_argument("--resume", default=None, help="checkpoint.pt to resume from")
    p.add_argument("--policy-only", default=None, help="skip CFR: build the final artefacts (policy net / iterate bank) from this checkpoint and evaluate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile for the network fits")
    return p


def resolve_args(args):
    """Fill preset-controlled hyperparameters that were not given explicitly."""
    preset = PRESETS[args.preset]
    explicit = {key: getattr(args, key, None) is not None for key in preset}
    for key, value in preset.items():
        if not explicit[key]:
            setattr(args, key, value)
    if explicit["policy_epochs"] and not explicit["policy_steps"]:
        args.policy_steps = None  # an explicit epoch count beats the preset's step count
    if args.policy_steps is None and args.policy_epochs is None:
        args.policy_epochs = PRESETS["default"]["policy_epochs"]
    paper_rules = args.preset == "paper" or getattr(args, "algo", None) in ("dream", "escher")
    if getattr(args, "rm_fallback", None) is None:  # DeepCFR / DREAM papers: the best action when no advantage is positive
        args.rm_fallback = "argmax" if paper_rules else "uniform"
    if getattr(args, "masked_loss", None) is None:
        args.masked_loss = getattr(args, "algo", None) in ("dream", "escher")
    if getattr(args, "net", None) is None:
        args.net = "deepcfr" if args.preset == "paper" else "current"
    if getattr(args, "loss_weights", None) is None:
        args.loss_weights = "paper"
    return args


RESUME_INHERITED = ("algo", "traversals", "adv_capacity", "strat_capacity", "value_steps", "batch_size", "policy_epochs",
                    "policy_steps", "policy_batch_size", "regret_power", "strategy_power", "epsilon", "q_steps", "q_batch",
                    "q_capacity", "value_trajectories", "eval_hands", "eval_every", "policy_eval_every", "lbr_every", "lbr_hands",
                    "lbr_final_hands", "lbr_tables", "lbr_model_iterates", "seed", "lr", "target_scale", "value_epsilon", "masked_loss",
                    "checkpoint_every", "loss_weights", "policy_lr_decay")
# what a checkpoint written before a hyperparameter existed was trained with (a resumed run continues as it began)
RESUME_LEGACY = {"loss_weights": "raw", "policy_lr_decay": 0.9}


def cli_args(argv=None):
    """The run's arguments: the command line, completed from the checkpoint (--resume / --policy-only: its
    hyperparameters unless given here, its directory unless --out is given, its network variant and game) and
    from the preset."""
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = args.policy_only or args.resume
    if checkpoint:
        probe = build_parser()  # which options are on this command line (also as --flag=value or as a prefix)
        for action in probe._actions:
            action.default = argparse.SUPPRESS
        given = set(vars(probe.parse_args(argv)))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True).get("args", {})
        for key in RESUME_INHERITED:
            if key in saved and key not in given:
                setattr(args, key, saved[key])
            elif key not in saved and key not in given and key in RESUME_LEGACY:
                setattr(args, key, RESUME_LEGACY[key])
        if "policy_epochs" in given and "policy_steps" not in given:
            args.policy_steps = None  # an explicit epoch count beats the checkpoint's step count (steps win when both are set)
        if "out" not in given:
            args.out = os.path.dirname(os.path.abspath(checkpoint))
    args = resolve_args(args)
    if checkpoint:  # the network variant and the game are fixed by the checkpoint
        cfg = DeepCFRTrainer.checkpoint_model_config(checkpoint)
        args.features, args.net, args.cards, args.dim, args.rm_fallback = (
            cfg["features"], cfg["arch"], cfg["cards"], cfg["dim"], cfg["rm_fallback"])
        game = GameConfig.from_dict(cfg["game"])
        args.bet_sizes, args.raise_cap = ",".join(str(s) for s in game.bet_sizes), game.raise_cap
        args.mask_redundant = "on" if game.mask_redundant else "off"
        args.game = "fhp" if game.limit and game.num_rounds == 2 else "hulh" if game.limit else "nlhe"
        args.game_config = game.to_dict()  # the name is a label; the trainer builds exactly this game
    return args


def main(argv=None):
    args = cli_args(argv)
    trainer = DeepCFRTrainer(args)
    if args.policy_only:
        trainer.load_checkpoint(args.policy_only)
        trainer.runner.close()
        trainer.finish()
        return
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
