"""Neural Fictitious Self-Play on the game protocol (Heinrich & Silver 2016, arXiv 1603.01121).

Per player: an action-value network ``Q`` (DQN with a target network) that learns a best response to the others'
behaviour from a circular memory ``M_RL`` of every transition, and an average-policy network ``Pi`` that imitates the
player's own past best responses from a reservoir ``M_SL``.  At the start of every hand each seat draws its mode -
eps-greedy on ``Q`` with probability ``eta``, else ``Pi`` - and keeps it for the hand; only best-response-mode
decisions go to ``M_SL``.  The result is the profile of the ``Pi`` networks.

Two presets (docs/superpowers/specs/2026-10-08-nfsp-design.md lists every difference between the sources):

* ``paper`` - the paper's Leduc setup: 1 x 64 ReLU MLPs, plain SGD (0.1 / 0.005), DQN, eps = 0.06 / sqrt(iteration);
* ``dream`` - the DREAM authors' NFSP (EricSteinberger/DREAM, Leduc_NFSP.py): Deep-CFR-style nets with a dueling Q
  head, Double DQN, SGD (0.1 / 0.01) with gradient clipping, eps = 0.06 / (1 + 0.01 sqrt(iteration)).

Rewards are in the DREAM code's unit (utilities / 2.6 in Leduc) in both: the paper does not state its unit, and with
antes its learning rate of 0.1 leaves half of the Q networks' hidden units dead (``--reward-scale 1``).
An iteration is 128 environment steps (one decision at each of 128 parallel tables) followed by 2 SGD steps per network.
Small games only: the tree is compiled to arrays, memories hold infoset indices, and the networks are numpy arrays
with a hand-written backward pass (an autograd step on a 64-unit network costs 5-7 times its arithmetic).

    python -m headsup.algos.nfsp --game leduc --preset paper --iterations 3000000 --json runs/x.json --checkpoint runs/x.pt
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from headsup.algos.best_response import TabularPolicy, exploitability
from headsup.algos.pdcfr import CHANCE, TERMINAL, Tree

# reward_scale None: the game's largest utility / 5 (the DREAM code divides by stack / 5: 2.6 in Leduc)
_PAPER = dict(arch="mlp", hidden=64, layers=1, lr_q=0.1, lr_pi=0.005, grad_clip=0.0, double_dqn=False, reward_scale=None,
              eps_start=0.06, eps_const=None, shared_explore=False, batch=128, updates=2, steps=128, envs=128,
              rl_capacity=200_000, sl_capacity=2_000_000, sl_min_prob=0.0, sl_window=False, target_every=300, eta=0.1)
PRESETS = {"paper": _PAPER,
           "dream": {**_PAPER, "arch": "deepcfr_dueling", "lr_pi": 0.01, "grad_clip": 1.0, "double_dqn": True, "eps_const": 0.01,
                     "shared_explore": True}}
ARCHS = ("mlp", "deepcfr_dueling")

_FILLED = {}


def _filled(value, *shape):
    """A cached float32 array of ``value``, never written to: numpy's elementwise loops and reductions are several
    times faster against an array of the operand's shape (and as a matrix product) than with a scalar or an axis."""
    key = (value, shape[1:])
    out = _FILLED.get(key)
    if out is None or len(out) < shape[0]:
        out = _FILLED[key] = np.full((max(shape[0], 256), *shape[1:]), value, dtype=np.float32)
    return out[: shape[0]]


# ----------------------------------------------------------------------------- the formulas
def epsilon(t, start=0.06, const=None):
    """Exploration during iteration t = 1, 2, ...: the paper's ``start / sqrt(t)`` or, with ``const``, the DREAM
    code's ``start / (1 + const * sqrt(t - 1))`` (its counter is the number of completed iterations)."""
    if const is None:
        return start / math.sqrt(t)
    return start / (1.0 + const * math.sqrt(t - 1))


def td_target(reward, done, next_target, next_legal, next_online=None):
    """The Q-learning target: the reward plus the TARGET network's value of the best legal next action (no discount);
    nothing is added when the hand has ended.  With ``next_online`` (Double DQN) the online network picks the next
    action and the target network values it."""
    pick = np.where(next_legal, next_target if next_online is None else next_online, -np.inf).argmax(1)
    return reward + (1.0 - done) * next_target[np.arange(len(pick)), pick]


def _softmax(logits, legal):
    """Softmax over the legal actions."""
    z = np.where(legal, logits, -np.inf)
    top = z[:, 0].copy()
    for k in range(1, z.shape[1]):
        np.maximum(top, z[:, k], out=top)
    z -= top[:, None]
    np.exp(z, out=z)
    z /= (z @ _filled(1.0, z.shape[1]))[:, None]
    return z


def cross_entropy(logits, legal, action):
    """Mean of ``-log softmax(logits)[action]`` with the softmax over the legal actions, and its gradient at the logits."""
    grad = _softmax(logits, legal)
    rows = np.arange(len(action))
    loss = -float(np.log(grad[rows, action]).sum()) / len(action)
    grad[rows, action] -= 1.0
    grad /= len(action)
    return loss, grad


# ----------------------------------------------------------------------------- memories
class Memory:
    """Named columns of a fixed capacity.  Circular by default (M_RL; the sliding-window ablation of M_SL).  With
    ``reservoir`` it is Vitter's Algorithm R: the m-th row offered replaces a uniformly drawn row with probability
    capacity / m, so the memory is a uniform sample of everything offered; ``min_prob`` floors that probability
    (the exponentially-averaging reservoir of the paper's Fig. 1b and of the DREAM code)."""

    def __init__(self, capacity, fields, rng, reservoir=False, min_prob=0.0):
        self.capacity, self.rng, self.reservoir, self.min_prob = int(capacity), rng, reservoir, min_prob
        self.data = {k: np.zeros(self.capacity, dtype=t) for k, t in fields}
        self.size = self.seen = 0

    def add(self, **cols):
        n = len(cols["info"])
        if n == 0:
            return
        start, keep = self.seen if self.reservoir else self.seen % self.capacity, None
        if start + n <= self.capacity:  # consecutive rows: a reservoir that is still filling, a FIFO that does not wrap
            slots = slice(start, start + n)
        elif not self.reservoir:
            slots = (start + np.arange(n)) % self.capacity
        else:  # r is uniform on [0, m): below the capacity with probability capacity / m, and then a uniform row
            m = self.seen + 1 + np.arange(n)
            r = self.rng.random(n) * (np.minimum(m, self.capacity / self.min_prob) if self.min_prob else m)
            keep = (r < self.capacity) | (m <= self.capacity)
            slots = np.where(m <= self.capacity, m - 1, r.astype(np.int64))[keep]
        for k, v in cols.items():  # of several rows drawn for one slot the last one stays (numpy assigns in order)
            self.data[k][slots] = v if keep is None or np.ndim(v) == 0 else v[keep]
        self.seen += n
        self.size = min(self.capacity, self.seen)

    def state_dict(self):
        return {"size": self.size, "seen": self.seen, **{k: v[: self.size].copy() for k, v in self.data.items()}}

    def load_state_dict(self, state):
        self.size, self.seen = int(state["size"]), int(state["seen"])
        for k, v in self.data.items():
            v[: self.size] = state[k]
        return self


# ----------------------------------------------------------------------------- networks
def mlp(in_dim, out_dim, hidden=64, layers=1):
    """The paper's network: fully-connected ReLU layers with PyTorch's default initialisation."""
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, out_dim))


def _forward(x, layers):
    """The activations of a stack of ReLU layers, the input first.  A layer: (weight, bias, their gradients)."""
    acts = [x]
    for W, b, _, _ in layers:
        x = x @ W.T
        x += b
        np.maximum(x, _filled(0.0, *x.shape), out=x)
        acts.append(x)
    return acts


def _backward_linear(d, layer, x, to_input=True):
    """y = x W^T + b: the layer's gradients from d = dL/dy; returns dL/dx."""
    W, _, gW, gb = layer
    np.matmul(d.T, x, out=gW)
    np.dot(_filled(1.0, len(d)), d, out=gb)
    return d @ W if to_input else None


def _backward(d, layers, acts, to_input=True):
    """Backward through `_forward`: ``d`` is the gradient at the stack's output; returns it at the input."""
    for i in range(len(layers) - 1, -1, -1):
        d = _backward_linear(d * (acts[i + 1] > 0), layers[i], acts[i], to_input or i > 0)
    return d


class _Net:
    """A torch module's parameters as numpy arrays - the same names and layout, views of one flat vector - with a
    hand-written backward pass: ``forward`` keeps the activations, ``backward`` turns the loss gradient at the output
    into the parameters' gradient, ``step`` is plain SGD.  The tests hold each subclass to its torch module."""

    def __init__(self, module):
        self.template = module
        self.theta = np.concatenate([p.detach().numpy().ravel() for p in module.parameters()]).astype(np.float32)
        self.grad = np.zeros_like(self.theta)
        self.w, self.g, pos = {}, {}, 0
        for k, p in module.named_parameters():
            self.w[k], self.g[k] = (v[pos : pos + p.numel()].reshape(tuple(p.shape)) for v in (self.theta, self.grad))
            pos += p.numel()

    def _layer(self, name):
        return self.w[name + ".weight"], self.w[name + ".bias"], self.g[name + ".weight"], self.g[name + ".bias"]

    def step(self, lr, clip=0.0):
        """One SGD step on the gradient of the last ``backward``; with ``clip`` the gradient is first scaled down to
        that norm (torch's clip_grad_norm_).  Returns the gradient's norm."""
        norm = math.sqrt(float(np.dot(self.grad, self.grad)))
        if clip:
            lr = lr * min(1.0, clip / (norm + 1e-6))
        self.grad *= lr
        self.theta -= self.grad
        return norm

    def state_dict(self):
        return {k: v.copy() for k, v in self.w.items()}

    def load_state_dict(self, state):
        for k, v in self.w.items():
            v[...] = state[k]
        return self

    def torch_module(self):
        """A torch module with the current weights."""
        module = copy.deepcopy(self.template)
        module.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.w.items()}, strict=False)
        return module


class MLP(_Net):
    def __init__(self, module):
        super().__init__(module)
        self.layers = [self._layer(str(i)) for i, m in enumerate(module) if isinstance(m, nn.Linear)]

    def forward(self, x, legal=None):
        self.acts = _forward(x, self.layers[:-1])
        W, b, _, _ = self.layers[-1]
        return self.acts[-1] @ W.T + b

    def backward(self, d):
        _backward(_backward_linear(d, self.layers[-1], self.acts[-1]), self.layers[:-1], self.acts, to_input=False)


class Dueling(_Net):
    """``game.make_model(arch="deepcfr_dueling")`` (headsup.games.leduc.DeepCFRNet with the DREAM authors' heads): card
    and bet branches, a trunk of three layers with skip connections, per-sample normalisation, then
    Q = (V + A - mean of A over the legal actions) * legal, or - ``policy`` - logits with illegal actions at -1e20."""

    EPS = 1e-5

    def __init__(self, module, policy):
        super().__init__(module)
        self.policy = policy
        self.split = len(module.card_idx)  # the card features come first in the infostate
        assert module.card_idx.tolist() == list(range(self.split))
        self.card, self.bet = [self._layer(f"card.{i}") for i in range(3)], [self._layer(f"bet.{i}") for i in range(2)]
        self.trunk = [[self._layer(f"trunk.{i}")] for i in range(3)]
        self.adv_layer, self.adv = [self._layer("adv_layer")], self._layer("adv")
        if not policy:
            self.v_layer, self.v = [self._layer("v_layer")], self._layer("v")

    def forward(self, x, legal):
        self.legal = legal = legal.astype(np.float32)
        self.c = _forward(np.ascontiguousarray(x[:, : self.split]), self.card)
        self.b = _forward(np.ascontiguousarray(x[:, self.split :]), self.bet)
        self.z0 = _forward(np.concatenate([self.c[-1], self.b[-1]], axis=1), self.trunk[0])
        self.z1 = _forward(self.z0[1], self.trunk[1])  # [z0, relu(W z0 + b)]; the next layer's input adds them
        self.z2 = _forward(self.z1[1] + self.z0[1], self.trunk[2])
        z = self.z2[1] + self.z2[0]
        dim = z.shape[1]
        self.u = u = z - (z @ _filled(1.0 / dim, dim))[:, None]
        self.s = s = np.sqrt(np.einsum("ij,ij->i", u, u) / (dim - 1))[:, None]  # torch's std: unbiased
        self.n = _forward(u / (s + self.EPS), self.adv_layer)
        W, b, _, _ = self.adv
        y = self.n[1] @ W.T + b
        if self.policy:
            return np.where(legal > 0, y, np.float32(-1e20))
        self.count = count = legal @ _filled(1.0, legal.shape[1])
        y *= legal
        y -= (y @ _filled(1.0, legal.shape[1]) / count)[:, None]
        self.hv = _forward(self.n[0], self.v_layer)
        W, b, _, _ = self.v
        y += self.hv[1] @ W.T + b
        y *= legal
        return y

    def backward(self, d):
        d = d * self.legal
        if self.policy:
            dn = _backward(_backward_linear(d, self.adv, self.n[1]), self.adv_layer, self.n)
        else:
            total = d @ _filled(1.0, d.shape[1])
            dn = _backward(_backward_linear(total[:, None], self.v, self.hv[1]), self.v_layer, self.hv)
            d -= (total / self.count)[:, None]
            d *= self.legal
            dn += _backward(_backward_linear(d, self.adv, self.n[1]), self.adv_layer, self.n)
        # n = u / (s + eps), s = sqrt(sum u^2 / (D - 1)), u = z - mean(z); a constant row (s = 0) gets no gradient
        u, s, dim = self.u, self.s, self.u.shape[1]
        dz = dn - (dn @ _filled(1.0 / dim, dim))[:, None]
        dz /= s + self.EPS
        dz -= u * (np.einsum("ij,ij->i", dn, u)[:, None] / ((s + self.EPS) ** 2 * (dim - 1) * np.maximum(s, 1e-12)))
        dz += _backward(dz, self.trunk[2], self.z2)  # z = relu(W x + b) + x: through the layer and around it
        dz += _backward(dz, self.trunk[1], self.z1)
        dz = _backward(dz, self.trunk[0], self.z0)
        k = self.c[-1].shape[1]
        _backward(np.ascontiguousarray(dz[:, :k]), self.card, self.c, to_input=False)
        _backward(np.ascontiguousarray(dz[:, k:]), self.bet, self.b, to_input=False)


def make_net(game, arch, hidden, layers, policy):
    """A numpy network initialised as its torch module: the paper's MLP, or the game's DREAM-style net (``layers`` is
    the MLP's depth only)."""
    if arch == "mlp":
        return MLP(mlp(game.obs_dim, game.num_actions, hidden, layers))
    if arch == "deepcfr_dueling":
        return Dueling(game.make_model(hidden=hidden, arch=arch, policy=policy), policy)
    raise ValueError(f"unknown arch {arch!r} (known: {', '.join(ARCHS)})")


# ----------------------------------------------------------------------------- the solver
RL_FIELDS = (("info", np.int32), ("action", np.int8), ("reward", np.float32), ("next", np.int32), ("done", np.float32))
SL_FIELDS = (("info", np.int32), ("action", np.int8))


def chance_closure(tree):
    """Deals in one draw: for every chance node the distribution over the first non-chance nodes below it (deals can
    follow each other).  Returns (row of each node, -1 for the others; nodes (rows, width); cumulative probabilities,
    padded above every uniform draw)."""
    def below(i):
        out = []
        for j, p in zip(tree.chance_child[i], tree.chance_prob[i]):
            if j >= 0:
                out += [(k, p * q) for k, q in below(j)] if tree.kind[j] == CHANCE else [(int(j), p)]
        return out

    dists = {int(i): below(i) for i in np.flatnonzero(tree.kind == CHANCE)}
    width = max(len(d) for d in dists.values())
    row = np.full(tree.num_nodes, -1, dtype=np.int64)
    nodes, cum = np.zeros((len(dists), width), dtype=np.int64), np.full((len(dists), width), 2.0)
    for r, (i, dist) in enumerate(dists.items()):
        row[i] = r
        nodes[r, : len(dist)] = [k for k, _ in dist]
        cum[r, : len(dist)] = np.cumsum([p for _, p in dist])
        cum[r, len(dist) - 1] = 1.0
    return row, nodes, cum


class NFSPSolver:
    """``preset`` gives the defaults (PRESETS), keywords override them; every setting is an attribute and in ``config``."""

    def __init__(self, game, preset="paper", seed=0, tree=None, **overrides):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r} (known: {', '.join(PRESETS)})")
        unknown = sorted(set(overrides) - set(PRESETS[preset]))
        if unknown:
            raise TypeError(f"unknown settings {', '.join(unknown)} (known: {', '.join(PRESETS[preset])})")
        self.game, self.preset = game, preset
        self.tree = tree = tree if tree is not None else Tree(game)
        self.config = cfg = {**PRESETS[preset], **overrides}
        if cfg["reward_scale"] is None:
            cfg["reward_scale"] = tree.max_utility / 5.0
        if cfg["steps"] % cfg["envs"]:
            raise ValueError(f"steps ({cfg['steps']}) must be a multiple of envs ({cfg['envs']}): every table makes one decision per pass")
        self.__dict__.update(cfg)  # self.lr_q, self.eta, ...
        self.rng = rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.Q = [make_net(game, self.arch, self.hidden, self.layers, False) for _ in range(2)]
        self.Pi = [make_net(game, self.arch, self.hidden, self.layers, True) for _ in range(2)]
        # the target networks, as their values at every infoset (the game is small: one pass per refit, none per minibatch)
        self.target_q = [None, None]
        self.rl_memory = [Memory(self.rl_capacity, RL_FIELDS, rng) for _ in range(2)]
        self.sl_memory = [Memory(self.sl_capacity, SL_FIELDS, rng, not self.sl_window, self.sl_min_prob) for _ in range(2)]
        self._chance = chance_closure(tree)
        self._utility = (tree.util / self.reward_scale).astype(np.float32)  # player 0's reward at the terminal nodes
        n = self.envs
        self._rows, self._batch_rows = np.arange(n), np.arange(self.batch)
        self.br = rng.random((n, 2)) < self.eta  # per table and seat: best-response mode in the current hand
        self.prev_info = np.full((n, 2), -1, dtype=np.int64)  # each seat's last decision in the current hand (-1: none yet)
        self.prev_action = np.zeros((n, 2), dtype=np.int64)
        self.node = self._deal(np.zeros(n, dtype=np.int64))  # every table waits at a decision node
        self.iteration = self.nodes_touched = self.episodes = 0
        self.q_updates = [0, 0]
        for p in (0, 1):
            self._sync_target(p)

    @property
    def epsilon(self):
        return epsilon(max(self.iteration, 1), self.eps_start, self.eps_const)

    # -- play ------------------------------------------------------------------------------------------
    def _deal(self, nodes):
        """Replace the chance nodes among ``nodes`` (the root: a new hand) by a sampled outcome, in place."""
        row, outcome, cum = self._chance
        at = row.take(nodes)
        c = (at >= 0).nonzero()[0]
        if len(c):
            at = at.take(c)
            nodes[c] = outcome[at, (self.rng.random(len(c))[:, None] >= cum.take(at, 0)).sum(1)]
        return nodes

    def _act(self, info, br, eps):
        """Actions at the infosets ``info`` (one per table; ``br``: the seat to act is in best-response mode there):
        eps-greedy on the player's own Q over the legal actions, or a sample of its average policy."""
        tree, rng = self.tree, self.rng
        legal, seat, obs = tree.info_legal.take(info, 0), tree.info_player.take(info), tree.info_obs.take(info, 0)
        probs = np.empty(legal.shape, dtype=np.float32)
        for p in (0, 1):
            mine = seat == p
            rows = (mine & br).nonzero()[0]
            if len(rows):
                lg = legal.take(rows, 0)
                q = np.where(lg, self.Q[p].forward(obs.take(rows, 0), lg), -np.inf)
                greedy = np.zeros(lg.shape, dtype=np.float32)
                greedy[np.arange(len(rows)), q.argmax(1)] = 1.0
                if self.shared_explore:  # the DREAM code: one coin for all of the seat's tables
                    random = rng.random() < eps
                    probs[rows] = lg if random else greedy
                else:
                    uniform = lg / (lg @ _filled(1.0, lg.shape[1]))[:, None]
                    probs[rows] = eps * uniform + (1.0 - eps) * greedy
            rows = (mine & ~br).nonzero()[0]
            if len(rows):
                lg = legal.take(rows, 0)
                probs[rows] = _softmax(self.Pi[p].forward(obs.take(rows, 0), lg), lg)
        cum = probs.cumsum(1)  # inverse CDF (rows need not sum to 1): an action with probability 0 is never hit
        return (rng.random(len(info))[:, None] * cum[:, -1:] >= cum).sum(1)

    def _play(self):
        """One decision at every table.  M_SL gets the best-response-mode decisions (exploratory ones included); M_RL
        gets every transition of either mode: a seat's previous decision is completed by its next infoset or, at the
        end of the hand, by its utility.  A finished table is dealt again and both seats draw their mode for the new
        hand.  Returns the tables whose hand ended."""
        tree, rows = self.tree, self._rows
        h = self.node
        info, seat = tree.info.take(h), tree.player.take(h)
        br = self.br[rows, seat]
        a = self._act(info, br, self.epsilon)
        for p in (0, 1):
            mine, prev = seat == p, self.prev_info[:, p]
            sel = (mine & br).nonzero()[0]
            self.sl_memory[p].add(info=info.take(sel), action=a.take(sel))
            sel = (mine & (prev >= 0)).nonzero()[0]
            self.rl_memory[p].add(info=prev.take(sel), action=self.prev_action[:, p].take(sel), reward=0.0, next=info.take(sel), done=0.0)
        self.prev_info[rows, seat], self.prev_action[rows, seat] = info, a
        nxt = tree.child[h, a]
        done = tree.kind.take(nxt) == TERMINAL
        ended = done.nonzero()[0]
        if len(ended):
            u = self._utility.take(nxt.take(ended))
            for p in (0, 1):
                has = (self.prev_info[:, p].take(ended) >= 0).nonzero()[0]
                sel = ended.take(has)
                self.rl_memory[p].add(info=self.prev_info[:, p].take(sel), action=self.prev_action[:, p].take(sel),
                                      reward=(u if p == 0 else -u).take(has), next=0, done=1.0)
            self.prev_info[ended] = -1
            self.br[ended] = self.rng.random((len(ended), 2)) < self.eta
            nxt[ended] = 0  # the root: dealt below
            self.episodes += len(ended)
        self.node = self._deal(nxt)
        return done

    # -- learning --------------------------------------------------------------------------------------
    def _batch(self, mem):
        return (self.rng.random(self.batch) * mem.size).astype(np.int64)  # uniform, with replacement

    def _sync_target(self, p):
        self.target_q[p] = self.Q[p].forward(self.tree.info_obs, self.tree.info_legal)

    def _update_q(self, p):
        """One SGD step of player p's Q on mean (Q(s, a) - target)^2 over a minibatch of M_RL."""
        mem, net, tree = self.rl_memory[p], self.Q[p], self.tree
        if mem.size < self.batch:
            return False
        j = self._batch(mem)
        info, action, nxt = mem.data["info"][j], mem.data["action"][j], mem.data["next"][j]
        legal = tree.info_legal.take(nxt, 0)
        # Double DQN asks the online network at the next states - before the pass whose activations backward() uses
        online = net.forward(tree.info_obs.take(nxt, 0), legal) if self.double_dqn else None
        target = td_target(mem.data["reward"][j], mem.data["done"][j], self.target_q[p].take(nxt, 0), legal, online)
        out = net.forward(tree.info_obs.take(info, 0), tree.info_legal.take(info, 0))
        rows = self._batch_rows
        grad = np.zeros(out.shape, dtype=np.float32)
        grad[rows, action] = (out[rows, action] - target) * (2.0 / self.batch)
        net.backward(grad)
        net.step(self.lr_q, self.grad_clip)
        self.q_updates[p] += 1
        if self.q_updates[p] % self.target_every == 0:
            self._sync_target(p)
        return True

    def _update_pi(self, p):
        """One SGD step of player p's Pi on mean -log Pi(a | s) over a minibatch of M_SL."""
        mem, net, tree = self.sl_memory[p], self.Pi[p], self.tree
        if mem.size < self.batch:
            return False
        j = self._batch(mem)
        info = mem.data["info"][j]
        legal = tree.info_legal.take(info, 0)
        net.backward(cross_entropy(net.forward(tree.info_obs.take(info, 0), legal), legal, mem.data["action"][j])[1])
        net.step(self.lr_pi, self.grad_clip)
        return True

    def iterate(self, n=1):
        for _ in range(n):
            self.iteration += 1
            for _ in range(self.steps // self.envs):
                self._play()
            self.nodes_touched += self.steps
            for _ in range(self.updates):
                for p in (0, 1):
                    self._update_q(p)
            for _ in range(self.updates):
                for p in (0, 1):
                    self._update_pi(p)
        return self

    # -- policies and evaluation -----------------------------------------------------------------------
    def average_policy(self):
        """The profile of the two average-policy networks - the algorithm's result."""
        tree = self.tree
        probs = np.zeros(tree.info_legal.shape)
        for p in (0, 1):
            mine = np.flatnonzero(tree.info_player == p)
            legal = tree.info_legal[mine]
            probs[mine] = _softmax(self.Pi[p].forward(tree.info_obs[mine], legal), legal)
        probs /= probs.sum(1, keepdims=True)
        return TabularPolicy(self.game, {k: probs[i] for i, k in enumerate(tree.info_keys)})

    def evaluate(self):
        return {"average": exploitability(self.game, self.average_policy())[0]}

    # -- checkpoints -----------------------------------------------------------------------------------
    def state_dict(self):
        """The whole state, copied: a solver that loads it continues exactly as this one would."""
        return {
            "game": self.game.name, "preset": self.preset, "config": dict(self.config),
            "iteration": self.iteration, "nodes_touched": self.nodes_touched, "episodes": self.episodes, "q_updates": list(self.q_updates),
            "Q": [n.state_dict() for n in self.Q], "Pi": [n.state_dict() for n in self.Pi], "target_q": [t.copy() for t in self.target_q],
            "rl_memory": [m.state_dict() for m in self.rl_memory], "sl_memory": [m.state_dict() for m in self.sl_memory],
            "node": self.node.copy(), "br": self.br.copy(), "prev_info": self.prev_info.copy(), "prev_action": self.prev_action.copy(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        mine, theirs = {"game": self.game.name, **self.config}, {"game": state["game"], **state["config"]}
        if mine != theirs:
            diff = ", ".join(f"{k} = {theirs.get(k)!r} (here: {mine.get(k)!r})" for k in mine if mine[k] != theirs.get(k))
            raise ValueError(f"the checkpoint was written with other settings: {diff}")
        self.iteration, self.nodes_touched, self.episodes = int(state["iteration"]), int(state["nodes_touched"]), int(state["episodes"])
        self.q_updates = list(state["q_updates"])
        for nets, key in ((self.Q, "Q"), (self.Pi, "Pi")):
            for net, saved in zip(nets, state[key]):
                net.load_state_dict(saved)
        self.target_q = [t.copy() for t in state["target_q"]]
        for mems, key in ((self.rl_memory, "rl_memory"), (self.sl_memory, "sl_memory")):
            for mem, saved in zip(mems, state[key]):
                mem.load_state_dict(saved)
        self.node, self.br = state["node"].copy(), state["br"].copy()
        self.prev_info, self.prev_action = state["prev_info"].copy(), state["prev_action"].copy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        return self


def eval_due(it, every, last):
    """Evaluations at 1, 2, 5, 10, 20, 50, ... iterations up to ``every``, then at its multiples, and at the end."""
    digits = str(it)
    return it == last or it % every == 0 or (it < every and digits[0] in "125" and not digits[1:].strip("0"))


def main(argv=None):
    from headsup.games import make_small_game

    def number_or(word):  # "--eps-const none", "--reward-scale auto"
        return lambda text: None if text == word else float(text)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc")
    p.add_argument("--preset", default="paper", choices=sorted(PRESETS), help="the defaults of the settings below")
    p.add_argument("--iterations", type=int, default=3_000_000, help="--steps decisions, then --updates SGD steps per network, each")
    s = p.add_argument_group("settings (default: the preset's)")
    s.add_argument("--arch", choices=ARCHS, default=argparse.SUPPRESS)
    s.add_argument("--hidden", type=int, default=argparse.SUPPRESS, help="units per layer")
    s.add_argument("--layers", type=int, default=argparse.SUPPRESS, help="hidden layers of the mlp")
    s.add_argument("--lr-q", type=float, default=argparse.SUPPRESS, help="plain SGD")
    s.add_argument("--lr-pi", type=float, default=argparse.SUPPRESS)
    s.add_argument("--grad-clip", type=float, default=argparse.SUPPRESS, help="gradient-norm clipping (0: none)")
    s.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                   help="the online network picks the next action, the target network values it")
    s.add_argument("--reward-scale", type=number_or("auto"), default=argparse.SUPPRESS,
                   help="utilities are divided by this ('auto': the largest utility / 5, the DREAM code's stack / 5)")
    s.add_argument("--eps-start", type=float, default=argparse.SUPPRESS)
    s.add_argument("--eps-const", type=number_or("none"), default=argparse.SUPPRESS,
                   help="'none': eps = start / sqrt(t) (the paper); c: start / (1 + c sqrt(t - 1)) (the DREAM code: 0.01)")
    s.add_argument("--shared-explore", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                   help="one exploration coin per seat and step for all tables (the DREAM code) instead of one per decision")
    s.add_argument("--batch", type=int, default=argparse.SUPPRESS)
    s.add_argument("--updates", type=int, default=argparse.SUPPRESS, help="SGD steps per network per iteration")
    s.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="decisions (of both players together) per iteration")
    s.add_argument("--envs", type=int, default=argparse.SUPPRESS, help="parallel tables; --steps must be a multiple")
    s.add_argument("--rl-capacity", type=int, default=argparse.SUPPRESS, help="M_RL, circular")
    s.add_argument("--sl-capacity", type=int, default=argparse.SUPPRESS, help="M_SL, a reservoir")
    s.add_argument("--sl-min-prob", type=float, default=argparse.SUPPRESS,
                   help="minimum insertion probability of M_SL (exponentially-averaging reservoir; the paper's LHE setup: 0.25)")
    s.add_argument("--sl-window", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                   help="M_SL as a sliding window (the paper's Fig. 1b ablation: it diverges)")
    s.add_argument("--target-every", type=int, default=argparse.SUPPRESS, help="Q updates between refits of the target network")
    s.add_argument("--eta", type=float, default=argparse.SUPPRESS, help="probability of best-response mode, drawn per hand and seat")
    p.add_argument("--eval-every", type=int, default=10_000,
                   help="exact exploitability at 1, 2, 5, 10, ... iterations up to this, then at its multiples")
    p.add_argument("--checkpoint", default=None, help="saved at evaluations; an existing file is resumed from")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0, help="at most one checkpoint per this many minutes (and at the end)")
    p.add_argument("--device", default="cpu", choices=["cpu"], help="the networks are numpy arrays with hand-written gradients")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    game = make_small_game(args.game)
    solver = NFSPSolver(game, args.preset, seed=args.seed, **{k: v for k, v in vars(args).items() if k in PRESETS[args.preset]})
    curve, elapsed = [], 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        solver.load_state_dict(saved["solver"])
        curve, elapsed = saved["curve"], saved["seconds"]
        print(f"resumed from {args.checkpoint} at iteration {solver.iteration}", flush=True)
    t0 = time.perf_counter() - elapsed
    last_save = time.perf_counter()

    def dump():
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"game": args.game, "algo": "nfsp", "preset": args.preset, "args": vars(args), "config": solver.config,
                           "curve": curve}, f, indent=2)

    for it in range(solver.iteration + 1, args.iterations + 1):
        solver.iterate()
        if eval_due(it, args.eval_every, args.iterations):
            ev = solver.evaluate()
            curve.append({"iteration": it, **ev, "nodes_touched": solver.nodes_touched, "env_steps": solver.nodes_touched,
                          "episodes": solver.episodes, "epsilon": solver.epsilon, "seconds": time.perf_counter() - t0})
            print(f"{args.game} nfsp {args.preset} it {it}: exploitability {ev['average']:.4f}  nodes {solver.nodes_touched:.3g}  "
                  f"episodes {solver.episodes}  eps {solver.epsilon:.2g}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            dump()
            due = time.perf_counter() - last_save >= 60.0 * args.checkpoint_minutes or it == args.iterations
            if args.checkpoint and due:  # atomic: a crash while writing leaves the previous checkpoint intact
                torch.save({"solver": solver.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}, args.checkpoint + ".tmp")
                os.replace(args.checkpoint + ".tmp", args.checkpoint)
                last_save = time.perf_counter()
    dump()


if __name__ == "__main__":
    main()
