from abc import ABC, abstractmethod
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openrl.envs.connect_env.utils import check_if_win


class BaseConnectEnv(gym.Env, ABC):
    def __init__(self, env_name):
        self.row, self.col = self._get_board_size()
        obs_space_dim = self.row * self.col
        self.action_size = self.row * self.col
        self.num_to_win = self._get_num2win()
        self.reward = 10
        self.env_name = env_name
        self.agent_num = 1
        self.player_num = 2

        obs_space_low = np.zeros(obs_space_dim) - 1e6
        obs_space_high = np.zeros(obs_space_dim) + 1e6
        obs_space_type = "float64"
        sobs_space_dim = obs_space_dim * self.agent_num
        sobs_space_low = np.zeros(sobs_space_dim) - 1e6
        sobs_space_high = np.zeros(sobs_space_dim) + 1e6

        if self.agent_num > 1:
            self.action_space = [
                spaces.Discrete(self.action_size) for _ in range(self.agent_num)
            ]
            self.observation_space = [
                spaces.Box(low=obs_space_low, high=obs_space_high, dtype=obs_space_type)
                for _ in range(self.agent_num)
            ]
            self.share_observation_space = [
                spaces.Box(
                    low=sobs_space_low, high=sobs_space_high, dtype=obs_space_type
                )
                for _ in range(self.agent_num)
            ]
        else:
            self.action_space = spaces.Discrete(self.action_size)
            self.observation_space = spaces.Box(
                low=obs_space_low, high=obs_space_high, dtype=obs_space_type
            )
            self.share_observation_space = spaces.Box(
                low=sobs_space_low, high=sobs_space_high, dtype=obs_space_type
            )

    @abstractmethod
    def _get_board_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def _get_num2win(self) -> int:
        raise NotImplementedError

    def step(self, action, is_enemy=True):
        # 传入action为0~8的数字
        row_pos, col_pos = action // self.col, action % self.col
        assert (
            self.state[row_pos][col_pos] == 0
        ), "({}, {}) pos has already be taken".format(row_pos, col_pos)
        self.state[row_pos][col_pos] = 2 if is_enemy else 1
        done, have_winner = False, False

        if check_if_win(self.state.copy(), row_pos, col_pos, self):
            done, have_winner = True, True
        if not done:
            if self.check_if_finish():
                done = True
        if done:
            if have_winner:
                reward = (-1) * self.reward if is_enemy else self.reward
                winner = "enemy" if is_enemy else "self"
            else:
                winner = "tie"
                reward = 0
        else:
            reward = 0
            winner = "no"
        info = {"who_win": winner}
        # print(self.state)
        return self.state.flatten().copy(), reward, done, False, info

    def check_if_finish(self):
        return (self.state == 0).sum() == 0

    def reset(self, seed=None, options=None, set_who_first=None):
        super().reset(seed=seed)
        # if seed is not None:
        #     self.seed(seed)
        self.state = np.zeros([self.row, self.col])  # 0无棋子，1我方棋子，2敌方棋子

        if set_who_first is not None:
            who_first = set_who_first
        else:
            if self.np_random.random() > 0.5:
                who_first = "enemy"
            else:
                who_first = "self"
        obs = self.state.flatten().copy()
        # return obs, {"who_first": who_first}
        # print(self.state)
        return obs, {"first": who_first == "self"}
