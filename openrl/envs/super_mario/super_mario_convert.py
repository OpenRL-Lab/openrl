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

from typing import Any, Dict, List, Optional, Union

import gym_super_mario_bros
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium import Wrapper
from nes_py.wrappers import JoypadSpace


class SuperMarioWrapper(Wrapper):
    def __init__(
        self,
        game: str,
        render_mode: Optional[Union[str, List[str]]] = None,
        disable_env_checker: Optional[bool] = None,
        **kwargs
    ):
        # unwrapped is used to adapt to higher versions of gym
        self.env = gym_super_mario_bros.make(game, **kwargs).unwrapped
        super().__init__(self.env)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        shape = self.env.observation_space.shape
        shape = (shape[2],) + shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=self.env.observation_space.dtype
        )

        self.action_space = gym.spaces.Discrete(self.env.action_space.n)

        self.env_name = game

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        obs = self.convert_observation(obs)

        return obs, reward, done, False, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        obs = self.env.reset()
        obs = self.convert_observation(obs)

        return obs, {}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def convert_observation(self, observation: np.array):
        obs = np.asarray(observation, dtype=np.uint8)
        obs = obs.transpose((2, 0, 1))

        return obs

    def render(self, **kwargs):
        image = self.env.render(mode="rgb_array")

        return image
