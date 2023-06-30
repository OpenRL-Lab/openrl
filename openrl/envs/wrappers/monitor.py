#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
import time
from typing import List, Union

import numpy as np

from openrl.envs.wrappers.base_wrapper import BaseWrapper


class Monitor(BaseWrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    """

    def __init__(self, env):
        super().__init__(env=env)
        self.t_start = time.time()

        self.rewards = []
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        self.total_steps = 0

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset.

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """

        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information or observation, reward, terminal, truncated, information
        """

        returns = self.env.step(action)
        if len(returns) == 4:
            done = returns[2]
        elif len(returns) == 5:
            done = returns[2] or returns[3]
        else:
            raise ValueError(
                "returns should have length 4 or 5, got length {}".format(len(returns))
            )
        # print("step", len(self.rewards), "rewards:", returns[1], "done:", done)

        self.rewards.append(returns[1])
        info = returns[-1]

        if np.all(done):
            ep_rew = np.sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)

            info["episode"] = ep_info
        self.total_steps += 1

        return *returns[:-1], info

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times
