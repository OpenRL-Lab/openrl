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

import pytest


@pytest.mark.unittest
def test_gridworld():
    from openrl.envs.common import make

    env = make("GridWorldEnv", env_num=6)
    obs, info = env.reset()
    obs, reward, done, info = env.step(env.random_action())
    env.close()


@pytest.mark.unittest
def test_gridworldrandom():
    from openrl.envs.common import make

    env = make("GridWorldEnvRandomGoal", env_num=6)
    obs, info = env.reset()
    obs, reward, done, info = env.step(env.random_action())
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
