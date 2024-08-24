import yaml
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_games.torch_runner import Runner


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self._model = model

    def forward(self, x):
        logits = self._model.a2c_network({"obs": self._model.norm_obs(x)})[0]
        probs = F.softmax(logits, dim=-1)
        return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="path to config")
    parser.add_argument(
        "-m", "--model", required=True, help="model checkpoint", type=str
    )

    args = vars(parser.parse_args())
    config_name = args["file"]
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
    runner = Runner()
    runner.load(config)

    agent = runner.create_player()
    agent.restore(args["model"])

    inputs = np.random.rand(1, 31).astype(np.float32)
    inputs = torch.tensor(inputs).cuda()
    torch.onnx.export(
        ModelWrapper(agent.model),
        inputs,
        "rl_games_model.onnx",
        input_names=["obs"],
        output_names=["logits"],
        opset_version=11,
    )


if __name__ == "__main__":
    main()
