from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Env:
    # create Connect3 environment from id
    if id == "connect3":
        env = Connect3Env(env_name=id, args=kwargs)

    return env


def check_if_win(state, check_row_pos, check_col_pos, all_args):
    def check_if_win_direction(now_state, direction, row_pos, col_pos, args):
        def check_if_valid(x_pos, y_pos):
            return (
                x_pos >= 0
                and x_pos <= (args["row"] - 1)
                and y_pos >= 0
                and y_pos <= (args["col"] - 1)
            )

        check_who = now_state[row_pos][col_pos]
        counting = 1
        bias_num = 1
        while True:
            new_row_pos = row_pos + bias_num * direction[0]
            new_col_pos = col_pos + bias_num * direction[1]
            if (
                not check_if_valid(new_row_pos, new_col_pos)
                or now_state[new_row_pos][new_col_pos] != check_who
            ):
                break
            else:
                counting += 1
                bias_num += 1
        bias_num = -1
        while True:
            new_row_pos = row_pos + bias_num * direction[0]
            new_col_pos = col_pos + bias_num * direction[1]
            if (
                not check_if_valid(new_row_pos, new_col_pos)
                or now_state[new_row_pos][new_col_pos] != check_who
            ):
                break
            else:
                counting += 1
                bias_num -= 1
        if counting >= args["num_to_win"]:
            return True
        else:
            return False

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横 竖 右下 右上
    for direction in directions:
        if check_if_win_direction(
            state, direction, check_row_pos, check_col_pos, all_args
        ):
            return True
    return False


class Connect3Env(gym.Env):
    def __init__(self, env_name, args):
        self.all_args = args
        self.row = args["row"]  # 行数
        self.col = args["col"]  # 列数
        obs_space_dim = self.row * self.col
        self.action_size = self.row * self.col
        self.num_to_win = args["num_to_win"]  # 需要连到这么多子才算赢
        self.reward = 10
        self.env_name = env_name
        self.agent_num = args["num_agents"]

        obs_space_low = np.zeros(obs_space_dim) - 1e6
        obs_space_high = np.zeros(obs_space_dim) + 1e6
        obs_space_type = "float64"
        sobs_space_dim = obs_space_dim * args["num_agents"]
        sobs_space_low = np.zeros(sobs_space_dim) - 1e6
        sobs_space_high = np.zeros(sobs_space_dim) + 1e6

        if args["num_agents"] > 1:
            self.action_space = [
                spaces.Discrete(self.action_size) for _ in range(args["num_agents"])
            ]
            self.observation_space = [
                spaces.Box(low=obs_space_low, high=obs_space_high, dtype=obs_space_type)
                for _ in range(args["num_agents"])
            ]
            self.share_observation_space = [
                spaces.Box(
                    low=sobs_space_low, high=sobs_space_high, dtype=obs_space_type
                )
                for _ in range(args["num_agents"])
            ]
        else:
            self.action_space = spaces.Discrete(self.action_size)
            self.observation_space = spaces.Box(
                low=obs_space_low, high=obs_space_high, dtype=obs_space_type
            )
            self.share_observation_space = spaces.Box(
                low=sobs_space_low, high=sobs_space_high, dtype=obs_space_type
            )

    def step(self, action, is_enemy=True):
        # 传入action为0~8的数字
        row_pos, col_pos = action // self.col, action % self.col
        assert (
            self.state[row_pos][col_pos] == 0
        ), "({}, {}) pos has already be taken".format(row_pos, col_pos)
        self.state[row_pos][col_pos] = 2 if is_enemy else 1
        done, have_winner = False, False

        if check_if_win(self.state.copy(), row_pos, col_pos, self.all_args):
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
        return self.state.flatten().copy(), reward, done, False, info

    def check_if_finish(self):
        return (self.state == 0).sum() == 0

    def reset(self, seed=None, options=None, set_who_first=None):
        self.state = np.zeros([self.row, self.col])  # 0无棋子，1我方棋子，2敌方棋子
        if set_who_first is not None:
            who_first = set_who_first
        else:
            if np.random.random() > 0.5:
                who_first = "enemy"
            else:
                who_first = "self"
        obs = self.state.flatten().copy()
        # return obs, {"who_first": who_first}
        return obs, {}

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def close(self):
        pass


if __name__ == "__main__":
    args = {"row": 3, "col": 3, "num_to_win": 3, "num_agents": 1}
    env = Connect3Env(env_name="connect3", args=args)
    obs, info = env.reset()
    obs, reward, done, _, info = env.step(1, is_enemy=True)
    env.close()
