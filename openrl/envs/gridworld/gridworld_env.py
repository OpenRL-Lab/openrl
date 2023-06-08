from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Env:
    # create GridWorld environment from id
    if id == "GridWorldEnv":
        env = GridWorldEnv(env_name=id, nrow=10, ncol=10)
    elif id == "GridWorldEnvRandomGoal":
        env = GridWorldEnv(env_name=id, nrow=10, ncol=10)
    return env


class GridWorldEnv(gym.Env):
    def __init__(self, env_name, nrow=5, ncol=5):
        self.env_name = env_name
        self.nrow = nrow
        self.ncol = ncol
        self.goal = np.array([1, 1])
        self.curr_pos = np.array([0, 0])
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.nrow - 1, self.ncol - 1, self.nrow - 1, self.ncol - 1]),
            dtype=int,
        )  # current position and target position
        self.action_space = spaces.Discrete(
            5
        )  # action [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
        self.steps = 0

    def step(self, action):
        if action == 0:  # stay
            pass
        elif action == 1:  # left
            self.curr_pos[0] -= 1
        elif action == 2:  # right
            self.curr_pos[0] += 1
        elif action == 3:  # up
            self.curr_pos[1] -= 1
        elif action == 4:  # down
            self.curr_pos[1] += 1
        else:
            raise ValueError("Invalid action!")

        self.curr_pos = np.clip(
            self.curr_pos,
            a_min=np.array([0, 0]),
            a_max=np.array([self.nrow - 1, self.ncol - 1]),
        )

        obs = np.concatenate((self.curr_pos, self.goal))
        reward = 0
        done = False
        if (self.curr_pos == self.goal).all():
            reward += 10
            done = True
            # print("Success!!!")
        else:
            reward -= 1

        if self.steps == 100:
            done = True
            reward -= 10
        else:
            self.steps += 1

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.steps = 0
        while True:
            self.curr_pos = np.random.randint(low=[0, 0], high=[self.nrow, self.ncol])
            if not (self.curr_pos == self.goal).all():
                obs = np.concatenate((self.curr_pos, self.goal))
                return obs, {}

    def render(self, mode="human"):
        pass


class GridWorldEnvRandomGoal(GridWorldEnv):
    def __init__(self, env_name, nrow=5, ncol=5):
        super(GridWorldEnvRandomGoal, self).__init__(env_name, nrow, ncol)

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.goal = np.random.randint(low=[0, 0], high=[self.nrow, self.ncol])
        while True:
            self.curr_pos = np.random.randint(low=[0, 0], high=[self.nrow, self.ncol])
            if not (self.curr_pos == self.goal).all():
                obs = np.concatenate((self.curr_pos, self.goal))
                return obs, {}


if __name__ == "__main__":
    env = GridWorldEnv(env_name="GridWorldEnv")
    obs, _ = env.reset(seed=0)
    print(env.curr_pos)
    while True:
        action = np.random.randint(0, 5)
        obs, reward, done, _, info = env.step(action)
        print("action: ", action)
        print("obs: ", obs, "reward: ", reward, "done: ", done)
        if done:
            break
