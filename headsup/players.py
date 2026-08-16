"""Batched players.

A player is a callable ``player(obs_batch: float32[N, 31]) -> int64[N]`` returning one
action per row.  Batching lets one network forward pass serve many tables at once (see
:class:`headsup.env.PokerVecEnv`).  Players may expose ``last_probs`` (float32[N, 4]) for
visualisation.
"""

import numpy as np

from headsup.enums import NUM_ACTIONS, Action


class RandomPlayer:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.integers(NUM_ACTIONS, size=len(obs))


class AlwaysCallPlayer:
    def __call__(self, obs):
        return np.full(len(obs), int(Action.CHECK_CALL), dtype=np.int64)


class AlwaysAllInPlayer:
    def __call__(self, obs):
        return np.full(len(obs), int(Action.ALL_IN), dtype=np.int64)


class AlwaysRaisePlayer:
    def __call__(self, obs):
        return np.full(len(obs), int(Action.RAISE), dtype=np.int64)


def sample_actions(probs, rng, deterministic=False):
    """Sample one action per row of a (N, A) probability matrix."""
    if deterministic:
        return probs.argmax(axis=1)
    cum = np.cumsum(probs, axis=1)
    u = rng.random((len(probs), 1)) * cum[:, -1:]
    return np.minimum((u > cum).sum(axis=1), probs.shape[1] - 1)


class TorchPolicyPlayer:
    """Samples from softmax(model(obs)).  ``model`` is a :class:`headsup.model.BaseModel`."""

    def __init__(self, model, device=None, deterministic=False, seed=None):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.device = device if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
        return probs.float().cpu().numpy()

    def __call__(self, obs):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng, self.deterministic)


class RegretMatchingPlayer:
    """Current CFR iterate: regret matching over the advantage nets (one net per seat).

    The seat is recovered from the observation (index 22 = 1 for the big blind), so one
    player object can sit in either seat.
    """

    def __init__(self, nets, device=None, seed=None):
        import torch

        self.torch = torch
        self.nets = [n.eval() for n in nets]
        self.device = device if device is not None else next(nets[0].parameters()).device
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs):
        torch = self.torch
        obs = np.asarray(obs, dtype=np.float32)
        out = np.empty((len(obs), NUM_ACTIONS), dtype=np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs).to(self.device)
            seat = x[:, 22].long()
            for s, net in enumerate(self.nets):
                mask = seat == s
                if mask.any():
                    adv = net(x[mask]).clamp(min=0)
                    total = adv.sum(dim=1, keepdim=True)
                    p = torch.where(total > 1e-6, adv / total.clamp(min=1e-6), torch.full_like(adv, 1.0 / NUM_ACTIONS))
                    out[mask.cpu().numpy()] = p.float().cpu().numpy()
        return out

    def __call__(self, obs):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng)


class NumpyPolicyPlayer:
    """Same as :class:`TorchPolicyPlayer` but on the numpy mirror (CPU workers)."""

    def __init__(self, numpy_model, deterministic=False, seed=None):
        self.model = numpy_model
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs):
        logits = self.model(np.asarray(obs, dtype=np.float32))
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def __call__(self, obs):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng, self.deterministic)


class ONNXPolicyPlayer:
    """Runs an exported rl_games exploiter (outputs action probabilities)."""

    def __init__(self, path, deterministic=False, seed=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        # models exported without dynamic axes have a fixed batch dimension
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        if self.fixed_batch is None or len(obs) == self.fixed_batch:
            return self.session.run([self.output_name], {self.input_name: obs})[0]
        out = np.empty((len(obs), NUM_ACTIONS), dtype=np.float32)
        b = self.fixed_batch
        for i in range(0, len(obs), b):
            chunk = obs[i : i + b]
            if len(chunk) < b:  # pad the last chunk
                chunk = np.concatenate([chunk, np.zeros((b - len(chunk), obs.shape[1]), np.float32)])
            out[i : i + b] = self.session.run([self.output_name], {self.input_name: chunk})[0][: len(out) - i]
        return out

    def __call__(self, obs):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng, self.deterministic)


SIMPLE_PLAYERS = {
    "random": RandomPlayer,
    "call": AlwaysCallPlayer,
    "allin": AlwaysAllInPlayer,
    "raise": AlwaysRaisePlayer,
}


def make_player(spec: str, device=None, deterministic=False, seed=None):
    """Build a player from a CLI spec.

    ``random`` | ``call`` | ``allin`` | ``raise`` | ``cfr[:path.pth]`` | ``onnx[:path.onnx]``
    """
    from headsup.paths import DEFAULT_ONNX_PATH, DEFAULT_POLICY_PATH

    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind in SIMPLE_PLAYERS:
        cls = SIMPLE_PLAYERS[kind]
        return cls(seed=seed) if kind == "random" else cls()
    if kind in ("cfr", "torch", "policy"):
        from headsup.device import get_device
        from headsup.model import load_model

        device = get_device(device) if not hasattr(device, "type") else device
        model = load_model(arg or DEFAULT_POLICY_PATH, device=device)
        return TorchPolicyPlayer(model, device=device, deterministic=deterministic, seed=seed)
    if kind == "onnx":
        return ONNXPolicyPlayer(arg or DEFAULT_ONNX_PATH, deterministic=deterministic, seed=seed)
    raise ValueError(f"unknown player spec {spec!r}")
