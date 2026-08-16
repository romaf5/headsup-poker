"""Export a trained rl_games exploiter to ONNX (input obs float32[N, 31] -> action probs).

    python rl_games_onnx.py -f rl_config/poker_env.yaml -m runs/<exp>/nn/exploitability.pth -o models/rl_games_exploiter.onnx
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import rl_games_env  # noqa: F401
from headsup.engine import OBS_DIM
from rl_games.torch_runner import Runner


class ModelWrapper(nn.Module):
    """obs -> action probabilities.  Input normalisation is re-implemented with float32
    constants (rl_games keeps float64 running stats in a scripted module, which neither the
    ONNX tracer nor MPS accept)."""

    def __init__(self, model):
        super().__init__()
        self.net = model.a2c_network
        self.normalize = bool(getattr(model, "normalize_input", False))
        if self.normalize:
            rms = model.running_mean_std
            self.register_buffer("mean", rms.running_mean.detach().float().clone())
            self.register_buffer("std", torch.sqrt(rms.running_var.detach().float() + float(rms.epsilon)))

    def forward(self, x):
        if self.normalize:
            x = torch.clamp((x - self.mean) / self.std, -5.0, 5.0)
        logits = self.net({"obs": x})[0]
        return F.softmax(logits, dim=-1)


def export(config_path, checkpoint, output, opponent=None):
    """Export an rl_games checkpoint to ONNX; returns the max abs diff vs the torch model."""
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    cfg = config["params"]["config"]
    cfg["device"] = cfg["device_name"] = "cpu"
    cfg["env_name"] = rl_games_env.ENV_NAME
    cfg.setdefault("env_config", {})
    cfg["env_config"]["opponent"] = opponent or "call"  # only needed to build the (unused) env
    runner = Runner()
    runner.load(config)
    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.model.eval()

    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        ModelWrapper(agent.model),
        dummy,
        output,
        input_names=["obs"],
        output_names=["probs"],
        dynamic_axes={"obs": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
    x = np.random.rand(3, OBS_DIM).astype(np.float32)
    with torch.no_grad():
        ref = torch.softmax(agent.model.a2c_network({"obs": agent.model.norm_obs(torch.from_numpy(x))})[0], dim=-1).numpy()
    out = sess.run(None, {"obs": x})[0]
    return float(np.abs(out - ref).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--file", required=True, help="path to config")
    parser.add_argument("-m", "--model", required=True, help="rl_games checkpoint (.pth)")
    parser.add_argument("-o", "--output", default="models/rl_games_exploiter.onnx")
    args = parser.parse_args()
    diff = export(args.file, args.model, args.output)
    print(f"exported {args.output}; max abs diff vs torch: {diff:.2e}")


if __name__ == "__main__":
    main()
