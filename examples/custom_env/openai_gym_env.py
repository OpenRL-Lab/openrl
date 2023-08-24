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

from typing import Any, Dict, Optional

import gym
from gym import spaces
from gym.envs.registration import EnvSpec, register
from gym.utils import seeding
from train_and_test import train_and_test

from openrl.envs.common import make


class IdentityEnv(gym.Env):
    spec = EnvSpec("IdentityEnv-v1")

    def __init__(self, **kwargs):
        self.dim = 2
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.dim)
        self.ep_length = 5
        self.current_step = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.generate_state()
        return self.state

    def step(self, action):
        reward = 1
        self.generate_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def generate_state(self) -> None:
        self.state = [self._np_random.randint(0, self.dim - 1)]

    def render(self, mode: str = "human") -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def close(self):
        pass


register(
    id="Custom_Env/IdentityEnv-v1",
    entry_point="openai_gym_env:IdentityEnv",
)

env = make("GymV21Environment-v0:Custom_Env/IdentityEnv-v1", env_num=10)

train_and_test(env)
