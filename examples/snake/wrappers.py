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
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openrl.envs.wrappers.base_wrapper import BaseObservationWrapper


def raw2vec(raw_obs, n_player=2):
    control_index = raw_obs["controlled_snake_index"][0]

    width = raw_obs["board_width"][0]
    height = raw_obs["board_height"][0]
    beans = raw_obs[1][0]

    ally_pos = raw_obs[control_index][0]
    enemy_pos = raw_obs[5 - control_index][0]

    obs = np.zeros(width * height * n_player, dtype=int)

    ally_head_h, ally_head_w = ally_pos[0]
    enemy_head_h, enemy_head_w = enemy_pos[0]
    obs[ally_head_h * width + ally_head_w] = 2
    obs[height * width + ally_head_h * width + ally_head_w] = 4
    obs[enemy_head_h * width + enemy_head_w] = 4
    obs[height * width + enemy_head_h * width + enemy_head_w] = 2

    for bean in beans:
        h, w = bean
        obs[h * width + w] = 1
        obs[height * width + h * width + w] = 1

    for p in ally_pos[1:]:
        h, w = p
        obs[h * width + w] = 3
        obs[height * width + h * width + w] = 5

    for p in enemy_pos[1:]:
        h, w = p
        obs[h * width + w] = 5
        obs[height * width + h * width + w] = 3

    obs_ = np.array([])
    for i in obs:
        obs_ = np.concatenate([obs_, np.eye(6)[i]])
    obs_ = obs_.reshape(-1, width * height * n_player * 6)

    return obs_


class ConvertObs(BaseObservationWrapper):
    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        BaseObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(576,), dtype=np.float32
        )

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """

        return raw2vec(observation)
