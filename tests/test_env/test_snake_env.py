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
import os
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from openrl.envs.common import make
from openrl.envs.wrappers.base_wrapper import BaseObservationWrapper
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


class ConvertObs(BaseObservationWrapper):
    def __init__(self, env: gym.Env):
        BaseObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(576,), dtype=np.float32
        )

    def observation(self, observation):
        new_obs = np.zeros((len(observation), 576), dtype=int)
        return new_obs


@pytest.mark.unittest
def test_snake():
    env_num = 2
    for i in [1, 3]:
        env = make(
            f"snakes_{i}v{i}",
            env_num=env_num,
            asynchronous=False,
            opponent_wrappers=[RandomOpponentWrapper],
            env_wrappers=[ConvertObs],
            auto_reset=False,
        )
        ep_num = 3
        for ep_now in range(ep_num):
            obs, info = env.reset()
            done = False
            step = 0

            while not np.any(done):
                obs, r, done, info = env.step(env.random_action())
                step += 1

        env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
