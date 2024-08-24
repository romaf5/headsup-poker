from poker_env import HeadsUpPoker, Action
from deepcfr.obs_processor import ObsProcessor

import torch
import numpy as np
from gym import spaces


class RandomPlayer:
    def __call__(self, _):
        return np.random.choice(
            [Action.FOLD, Action.CHECK_CALL, Action.RAISE, Action.ALL_IN]
        )


class RLGamesObsProcessor(ObsProcessor):
    def __call__(self, obs):
        board = self._process_board(obs["board"])
        player_hand = self._process_hand(obs["player_hand"])
        stage = self._process_stage(obs["stage"])
        first_to_act_next_stage = self._process_first_to_act_next_stage(
            obs["first_to_act_next_stage"]
        )
        bets_and_stacks = self._process_bets_and_stacks(obs)
        return np.array(
            player_hand + board + [stage, first_to_act_next_stage] + bets_and_stacks,
            dtype=np.float32,
        )


class PolicyPlayerWrapper:
    def __init__(self, policy):
        self.policy = policy

    def _batch_obses(self, obses):
        return {k: torch.tensor([obs[k] for obs in obses]) for k in obses[0].keys()}

    def __call__(self, obs):
        with torch.no_grad():
            obs_dict = {
                "board_and_hand": [int(x) for x in obs[:21]],
                "stage": int(obs[21]),
                "first_to_act_next_stage": int(obs[22]),
                "bets_and_stacks": list(obs[23:]),
            }

            obs = self._batch_obses([obs_dict])
            action_distribution = self.policy(obs)[0]
            action_distribution = torch.nn.functional.softmax(
                action_distribution, dim=-1
            )
            action = torch.multinomial(action_distribution, 1).item()
            return action


class HeadsUpPokerRLGames(HeadsUpPoker):
    def __init__(self):
        from deepcfr.model import BaseModel

        obs_processor = RLGamesObsProcessor()
        policy = BaseModel()
        policy.load_state_dict(
            torch.load(
                "deepcfr/policy.pth",
                weights_only=True,
                map_location="cpu",
            )
        )
        model = PolicyPlayerWrapper(policy)
        # model = RandomPlayer()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        super(HeadsUpPokerRLGames, self).__init__(obs_processor, model)


if __name__ == "__main__":
    env = HeadsUpPokerRLGames()
    observation = env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()
