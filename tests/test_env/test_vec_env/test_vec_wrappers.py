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
from openrl.envs.vec_env.wrappers.zero_reward_wrapper import ZeroRewardWrapper


@pytest.mark.unittest
def test_zero_reward_wrapper():
    env = make("IdentityEnv", env_num=1)
    env = ZeroRewardWrapper(env)
    env.reset(seed=0)
    while True:
        obs, reward, done, info = env.step(env.random_action())
        assert np.all(reward == 0), "reward should be zero"
        if done:
            break
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
