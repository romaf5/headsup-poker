import torch
import numpy as np

from model import BaseModel
from player_wrapper import PolicyPlayerWrapper
from pokerenv_crf import HeadsUpPoker, ObsProcessor
from simple_players import RandomPlayer, AlwaysCallPlayer, AlwaysAllInPlayer


class EvalPolicyPlayer:
    def __init__(self, env):
        self.env = env
        self.opponent_players = {
            "random": RandomPlayer(),
            "call": AlwaysCallPlayer(),
            "allin": AlwaysAllInPlayer(),
        }

    def eval(self, player, games_to_play=50000):
        scores = {}
        for opponent_name, opponent_player in self.opponent_players.items():
            rewards = []
            for play_as_idx in [0, 1]:
                for _ in range(games_to_play):
                    obs = self.env.reset()
                    done = False
                    while not done:
                        if obs["player_idx"] == play_as_idx:
                            action = player(obs)
                        else:
                            action = opponent_player(obs)
                        obs, reward, done, _ = self.env.step(action)
                        if done:
                            rewards.append(reward[play_as_idx])
            scores[opponent_name] = np.mean(rewards)
        return scores


if __name__ == "__main__":
    env = HeadsUpPoker(ObsProcessor())

    model = BaseModel().cuda()
    model.load_state_dict(torch.load("policy.pth"))
    model.eval()

    player = PolicyPlayerWrapper(model)
    evaluator = EvalPolicyPlayer(env)
    scores = evaluator.eval(player)
    print(scores)
