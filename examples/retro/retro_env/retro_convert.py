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

import gymnasium
import numpy as np
from gymnasium import Wrapper
from retro import RetroEnv


class CustomRetroEnv(RetroEnv):
    def __init__(self, game: str, **kwargs):
        super(CustomRetroEnv, self).__init__(game, **kwargs)

    def seed(self, seed: Optional[int] = None):
        seed1 = np.random.seed(seed)

        seed1 = np.random.randint(0, 2**31)
        seed2 = np.random.randint(0, 2**31)

        return [seed1, seed2]

    def render(self, mode: Optional[str] = "human", close: Optional[bool] = False):
        if close:
            if self.viewer:
                self.viewer.close()
            return

        img = self.get_screen() if self.img is None else self.img

        return img


class RetroWrapper(Wrapper):
    def __init__(
        self,
        game: str,
        render_mode: Optional[Union[str, List[str]]] = None,
        disable_env_checker: Optional[bool] = None,
        **kwargs
    ):
        self.env = CustomRetroEnv(game=game, **kwargs)

        super().__init__(self.env)

        shape = self.env.observation_space.shape
        shape = (shape[2],) + shape[0:2]
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=self.env.observation_space.dtype,
        )

        self.action_space = gymnasium.spaces.Discrete(self.env.action_space.n)

        self.env_name = game

    def step(self, action_index: int):
        action = self.convert_action(action_index)
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

    def convert_action(self, action_index: int):
        action = [0] * self.env.action_space.n
        action[action_index] = 1

        return action

    def render(self, **kwargs):
        image = self.env.get_screen()

        return image
