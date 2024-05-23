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

import crafter
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper


class CrafterWrapper(Wrapper):
    def __init__(
        self,
        name: str,
        render_mode: Optional[Union[str, List[str]]] = None,
        disable_env_checker: Optional[bool] = None,
        **kwargs
    ):
        self.env_name = name

        self.env = crafter.Env()
        self.env = crafter.Recorder(
            self.env,
            "run_results/crafter_traj",
            save_stats=False,  # if True, save the stats of the environment to example/crafter/crafter_traj
            save_episode=False,
            save_video=False,
        )

        super().__init__(self.env)

        shape = self.env.observation_space.shape
        shape = (shape[2],) + shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=self.env.observation_space.dtype
        )

        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        
        self.rand_seed = 42

    def step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.convert_observation(obs)

        return obs, reward, done, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        
        if seed is not None:
            self.rand_seed = seed
        else:
            self.rand_seed += 1
        
        obs, info = self.env.reset(self.rand_seed, options)
        obs = self.convert_observation(obs)

        return obs, info

    def convert_observation(self, observation: np.array):
        obs = np.asarray(observation, dtype=np.uint8)
        obs = obs.transpose((2, 0, 1))

        return obs
