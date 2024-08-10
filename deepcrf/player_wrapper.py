import torch
import numpy as np


def _batch_obses(obses):
    batch_obses = {}
    for k in obses[0].keys():
        batch_obses[k] = []
    for obs in obses:
        for k, v in obs.items():
            batch_obses[k].append(v)
    for k in batch_obses.keys():
        batch_obses[k] = torch.tensor(batch_obses[k]).cuda()
    return batch_obses


class PolicyPlayerWrapper:
    def __init__(self, player):
        self.player = player

    def __call__(self, obs):
        with torch.no_grad():
            obs = _batch_obses([obs])
            action_distribution = self.player(obs)
            action_distribution = torch.clamp(action_distribution, min=0)
            total_action_distribution = torch.sum(action_distribution)
            if total_action_distribution <= 0:
                return np.random.choice(range(len(action_distribution)))
            action_distribution /= total_action_distribution
            action = torch.multinomial(action_distribution, 1).item()
            return action
