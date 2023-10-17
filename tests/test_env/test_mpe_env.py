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

import numpy as np
import pytest

from openrl.envs.common import make


@pytest.mark.unittest
def test_mpe():
    env_num = 3
    env = make("simple_spread", env_num=env_num)
    obs, info = env.reset()
    obs, reward, done, info = env.step(env.random_action())
    assert env.agent_num == 3
    assert env.parallel_env_num == env_num
    env.close()


@pytest.mark.unittest
def test_mpe_render():
    render_model = "human"
    env_num = 2
    env = make(
        "simple_spread", render_mode=render_model, env_num=env_num, asynchronous=False
    )

    env.reset(seed=0)
    done = False
    step = 0
    total_reward = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.

        obs, r, done, info = env.step(env.random_action())
        step += 1
        total_reward += np.mean(r)

    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
