import torch
import numpy as np

from model import BaseModel
from enums import Action
from player_wrapper import PolicyPlayerWrapper
from deepcfr.pokerenv_cfr import HeadsUpPoker, ObsProcessor
from simple_players import RandomPlayer, AlwaysCallPlayer, AlwaysAllInPlayer
from cfr_env_wrapper import CFREnvWrapper
from copy import deepcopy


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


class EvalExploitabilityScore:
    def __init__(self, env, player):
        self.player = player
        self.env = CFREnvWrapper(env)

    def _play_game(self, env, play_as_idx):
        if env.done:
            return env.reward[play_as_idx]

        if env.obs["player_idx"] == play_as_idx:
            action = self.player(env.obs)
            env.step(action)
            return self._play_game(env, play_as_idx)

        rewards = []
        for action in Action:
            env_copy = deepcopy(env)
            env_copy.step(action)
            rewards.append(self._play_game(env_copy, play_as_idx))
        return min(rewards)

    def eval(self, games_to_play=50000):
        rewards = []
        for play_as_idx in range(2):
            for _ in range(games_to_play):
                self.env.reset()
                reward = self._play_game(self.env, play_as_idx)
                rewards.append(reward)
        return np.mean(rewards)


if __name__ == "__main__":
    env = HeadsUpPoker(ObsProcessor())

    model = BaseModel().cuda()
    model.load_state_dict(torch.load("policy.pth", weights_only=True))
    model.eval()

    player = PolicyPlayerWrapper(model)
    evaluator = EvalPolicyPlayer(env)
    scores = evaluator.eval(player)
    print("Average rewards against simple players\n", scores)

    exploitability_evaluator = EvalExploitabilityScore(env, player)
    exploitability_score = exploitability_evaluator.eval()
    print("Exploitability score", exploitability_score)
