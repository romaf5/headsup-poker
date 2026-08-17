"""Batched players.

A player is a callable ``player(obs_batch: float32[N, OBS_DIM], ids=None) -> int64[N]``
returning one action per row.  Batching lets one network forward pass serve many tables at
once (see :class:`headsup.env.PokerVecEnv`).  ``ids`` are the table indices of the rows;
stateless players ignore them, stateful ones (SD-CFR) use them to track per-table state.
Players may expose ``last_probs`` (float32[N, 4]) and ``probs(obs, ids=None)`` for
visualisation/advice.  Envs always hand out the full observation; networks that were trained
on fewer features (e.g. the 31-feature layout) read the prefix they need.
"""

import numpy as np

from headsup.enums import Action
from headsup.game import DEFAULT_GAME


class RandomPlayer:
    """Uniform over the legal actions of ``game``."""

    def __init__(self, seed=None, game=DEFAULT_GAME):
        self.rng = np.random.default_rng(seed)
        self.game = game
        self.last_probs = None

    def probs(self, obs, ids=None):
        n = self.game.num_actions
        return mask_illegal(np.full((len(obs), n), 1.0 / n, dtype=np.float32), obs, self.game)

    def __call__(self, obs, ids=None):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng)


class _FixedActionPlayer:
    def __init__(self, seed=None, game=DEFAULT_GAME):
        self.game = game
        self.action = self._action(game)
        self.last_probs = None

    @staticmethod
    def _action(game):
        return int(Action.CHECK_CALL)

    def probs(self, obs, ids=None):
        p = np.zeros((len(obs), self.game.num_actions), dtype=np.float32)
        p[:, self.action] = 1.0
        return p

    def __call__(self, obs, ids=None):
        self.last_probs = self.probs(obs)
        return np.full(len(obs), self.action, dtype=np.int64)


class AlwaysCallPlayer(_FixedActionPlayer):
    pass


class AlwaysAllInPlayer(_FixedActionPlayer):
    @staticmethod
    def _action(game):
        return game.all_in


class AlwaysRaisePlayer(_FixedActionPlayer):
    """Always the first (smallest) raise size."""

    @staticmethod
    def _action(game):
        return int(Action.RAISE)


def mask_illegal(probs, obs, game=DEFAULT_GAME):
    """Zero the probability of actions that are not legal in the observed state (FOLD when nothing
    is to call; with ``game.mask_redundant`` also duplicate raises) and renormalise (in place).
    Rows left without mass become uniform over the legal actions."""
    from headsup.engine import legal_mask_from_obs

    legal = legal_mask_from_obs(obs, game)
    if legal.shape[1] != probs.shape[1]:
        raise ValueError(f"player outputs {probs.shape[1]} actions but the game has {legal.shape[1]}")
    probs[~legal] = 0.0
    s = probs.sum(axis=1, keepdims=True)
    uniform = legal / legal.sum(axis=1, keepdims=True)
    probs[:] = np.where(s > 0, probs / np.maximum(s, 1e-12), uniform)
    return probs


def regret_matching_torch(adv, legal, fallback="uniform"):
    """Batched regret matching (torch): ``adv`` (..., N, A), ``legal`` bool (N, A).

    Illegal actions get probability 0; when no legal advantage is positive the ``fallback``
    plays uniform over the legal actions or the highest legal advantage (``argmax``).
    Mirrors ``regret_matching`` in headsup/cpp/headsup_cpp.cpp and headsup/deepcfr/traverse.py.
    """
    import torch

    adv = adv.clone()
    legal = (legal if torch.is_tensor(legal) else torch.as_tensor(np.asarray(legal))).to(device=adv.device, dtype=torch.bool)
    adv[..., ~legal] = -float("inf")
    pos = adv.clamp(min=0.0)
    total = pos.sum(dim=-1, keepdim=True)
    if fallback == "argmax":
        fb = torch.nn.functional.one_hot(adv.argmax(dim=-1), adv.shape[-1]).to(adv.dtype)
    else:
        allowed = legal.to(adv.dtype)
        fb = allowed / allowed.sum(dim=-1, keepdim=True)
    return torch.where(total > 1e-6, pos / total.clamp(min=1e-6), fb)


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
        self.game = model.game
        self.device = device if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs, ids=None):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
        return mask_illegal(probs.float().cpu().numpy(), obs, self.game)

    def __call__(self, obs, ids=None):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng, self.deterministic)


class RegretMatchingPlayer:
    """Current CFR iterate: regret matching over the advantage nets (one net per seat).

    The seat is recovered from the observation (index 22 = 1 for the big blind), so one
    player object can sit in either seat.  The regret-matching fallback (uniform / argmax)
    is the one the nets were trained with (``net.rm_fallback``).
    """

    def __init__(self, nets, device=None, seed=None):
        import torch

        self.torch = torch
        self.nets = [n.eval() for n in nets]
        self.game = nets[0].game
        self.device = device if device is not None else next(nets[0].parameters()).device
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs, ids=None):
        from headsup.engine import legal_mask_from_obs

        torch = self.torch
        obs = np.asarray(obs, dtype=np.float32)
        out = np.empty((len(obs), self.game.num_actions), dtype=np.float32)
        legal = torch.as_tensor(legal_mask_from_obs(obs, self.game), device=self.device)
        with torch.no_grad():
            x = torch.as_tensor(obs).to(self.device)
            seat = x[:, 22].long()
            for s, net in enumerate(self.nets):
                mask = seat == s
                if mask.any():
                    p = regret_matching_torch(net(x[mask]), legal[mask], getattr(net, "rm_fallback", "uniform"))
                    out[mask.cpu().numpy()] = p.float().cpu().numpy()
        return out

    def __call__(self, obs, ids=None):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng)


class NumpyPolicyPlayer:
    """Same as :class:`TorchPolicyPlayer` but on the numpy mirror (CPU workers)."""

    def __init__(self, numpy_model, deterministic=False, seed=None):
        self.model = numpy_model
        self.game = numpy_model.game
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs, ids=None):
        logits = self.model(np.asarray(obs, dtype=np.float32))
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return mask_illegal(p / p.sum(axis=1, keepdims=True), obs, self.game)

    def __call__(self, obs, ids=None):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng, self.deterministic)


class ONNXPolicyPlayer:
    """Runs an exported rl_games exploiter (outputs action probabilities)."""

    def __init__(self, path, deterministic=False, seed=None, game=DEFAULT_GAME):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        # models exported without dynamic axes have a fixed batch dimension
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.obs_dim = inp.shape[1] if isinstance(inp.shape[1], int) else None  # features the exploiter was trained on
        self.game = game
        if isinstance(out.shape[-1], int) and out.shape[-1] != game.num_actions:
            raise ValueError(f"{path} outputs {out.shape[-1]} actions, the game has {game.num_actions}")
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.last_probs = None

    def probs(self, obs, ids=None):
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_dim is not None:
            if obs.shape[1] < self.obs_dim:
                raise ValueError(f"observation has {obs.shape[1]} features, this exploiter needs {self.obs_dim}")
            obs = obs[:, : self.obs_dim]
        obs = np.ascontiguousarray(obs)
        if self.fixed_batch is None or len(obs) == self.fixed_batch:
            return self.session.run([self.output_name], {self.input_name: obs})[0]
        out = np.empty((len(obs), self.game.num_actions), dtype=np.float32)
        b = self.fixed_batch
        for i in range(0, len(obs), b):
            chunk = obs[i : i + b]
            if len(chunk) < b:  # pad the last chunk
                chunk = np.concatenate([chunk, np.zeros((b - len(chunk), obs.shape[1]), np.float32)])
            out[i : i + b] = self.session.run([self.output_name], {self.input_name: chunk})[0][: len(out) - i]
        return out

    def __call__(self, obs, ids=None):
        self.last_probs = self.probs(obs)
        return sample_actions(self.last_probs, self.rng, self.deterministic)


SIMPLE_PLAYERS = {
    "random": RandomPlayer,
    "call": AlwaysCallPlayer,
    "allin": AlwaysAllInPlayer,
    "raise": AlwaysRaisePlayer,
}


def parse_sdcfr_spec(arg):
    """``path[@exact|@sample][@g<gamma>][@t<N>][@k<K>]`` -> (path, mode, gamma, iterations, thin) with
    ``iterations`` (use the first N iterates) and ``thin`` (K representative iterates) None if absent."""
    parts = arg.split("@")
    path, opts = parts[0], parts[1:]
    mode = next((o for o in opts if o in ("exact", "sample")), "sample")
    gamma = next((float(o[1:]) for o in opts if o.startswith("g")), 1.0)
    iterations = next((int(o[1:]) for o in opts if o.startswith("t")), None)
    thin = next((int(o[1:]) for o in opts if o.startswith("k")), None)
    return path, mode, gamma, iterations, thin


def make_player(spec: str, device=None, deterministic=False, seed=None, game=None):
    """Build a player from a CLI spec.  ``game`` (:class:`headsup.game.GameConfig`) is the action
    tree for players that do not carry one themselves (simple bots, ONNX exploiters); network
    players bring their own (``player.game``).

    ``random`` | ``call`` | ``allin`` | ``raise`` | ``cfr[:path.pth]`` | ``onnx[:path.onnx]`` |
    ``sdcfr:path/iterates.pt[@exact][@g<gamma>][@t<N>][@k<K>]`` (Single Deep CFR average strategy; default
    trajectory sampling; ``@g2`` quadratic iterate weights; ``@t100`` = the average after 100 iterations;
    ``@k64`` = the bank thinned to 64 representative iterates) |
    ``iterate:path/iterates.pt[@t<N>]`` (the current strategy of iteration N - regret matching on that
    iteration's advantage nets; default: the last one)
    """
    from headsup.paths import DEFAULT_ONNX_PATH, DEFAULT_POLICY_PATH

    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    game = game or DEFAULT_GAME
    if kind in SIMPLE_PLAYERS:
        return SIMPLE_PLAYERS[kind](seed=seed, game=game)
    if kind in ("cfr", "torch", "policy"):
        from headsup.device import get_device
        from headsup.model import load_model

        device = get_device(device) if not hasattr(device, "type") else device
        model = load_model(arg or DEFAULT_POLICY_PATH, device=device)
        return TorchPolicyPlayer(model, device=device, deterministic=deterministic, seed=seed)
    if kind == "onnx":
        return ONNXPolicyPlayer(arg or DEFAULT_ONNX_PATH, deterministic=deterministic, seed=seed, game=game)
    if kind == "sdcfr":
        from headsup.device import get_device
        from headsup.sdcfr import SDCFRPlayer

        path, mode, gamma, iterations, thin = parse_sdcfr_spec(arg)
        device = get_device(device) if not hasattr(device, "type") else device
        return SDCFRPlayer.load(path, device=device, mode=mode, seed=seed, weight_power=gamma, iterations=iterations, thin=thin)
    if kind == "iterate":
        import torch

        from headsup.device import get_device
        from headsup.model import BaseModel

        path, _, _, iterations, _ = parse_sdcfr_spec(arg)
        device = get_device(device) if not hasattr(device, "type") else device
        data = torch.load(path, map_location="cpu", weights_only=True)
        t = data["T"] - 1 if iterations is None else min(iterations, data["T"] - 1)
        nets = []
        for seat in (0, 1):
            net = BaseModel(config=data["config"])
            net.load_state_dict({k: v[t] for k, v in data["seats"][seat].items()})
            nets.append(net.to(device).eval())
        return RegretMatchingPlayer(nets, device=device, seed=seed)
    raise ValueError(f"unknown player spec {spec!r}")
