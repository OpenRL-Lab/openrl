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
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from openrl.datasets.expert_dataset import ExpertDataset


class OfflineEnv(gym.Env):
    _np_random: Optional[np.random.Generator] = None
    env_name = "OfflineEnv"

    def __init__(self, dataset_path, env_id: int, env_num: int, seed: int):
        self.dataset = ExpertDataset(
            dataset_path, env_id=env_id, env_num=env_num, seed=seed
        )
        self.observation_space = self.dataset.observation_space
        self.action_space = self.dataset.action_space
        self.agent_num = self.dataset.agent_num
        self.traj_num = len(self.dataset.trajectories["episode_lengths"])
        self.traj_index = None
        self.traj_length = None
        self.step_index = None

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        if self.traj_index is None:
            self.traj_index = 0
        else:
            self.traj_index += 1
            self.traj_index %= self.traj_num
        self.traj_length = self.dataset.trajectories["episode_lengths"][self.traj_index]
        assert (
            self.traj_length
            == len(self.dataset.trajectories["obs"][self.traj_index]) - 1
        )
        assert self.traj_length == len(
            self.dataset.trajectories["action"][self.traj_index]
        )
        self.step_index = 0
        return (
            self.dataset.trajectories["obs"][self.traj_index][self.step_index],
            self.dataset.trajectories["info"][self.traj_index][self.step_index],
        )

    def step(self, action):
        obs = self.dataset.trajectories["obs"][self.traj_index][self.step_index + 1]
        reward = self.dataset.trajectories["reward"][self.traj_index][self.step_index]
        action = self.dataset.trajectories["action"][self.traj_index][self.step_index]
        done = self.dataset.trajectories["done"][self.traj_index][self.step_index]
        info = self.dataset.trajectories["info"][self.traj_index][self.step_index]
        if isinstance(info, list):
            info = {"info": info}
        info.update({"data_action": action})

        self.step_index += 1

        return obs, reward, done, info

    def close(self):
        pass
