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

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env.wrappers.gen_data import GenDataWrapper
from openrl.envs.wrappers.monitor import Monitor

env_wrappers = [
    Monitor,
]


def gen_data(total_episode, data_save_path):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.

    env = make(
        "IdentityEnv",
        env_num=1,
        asynchronous=True,
        env_wrappers=env_wrappers,
    )

    env = GenDataWrapper(
        env, data_save_path=data_save_path, total_episode=total_episode
    )
    env.reset()
    done = False
    ep_length = 0
    while not done:
        obs, r, done, info = env.step(env.random_action())
        ep_length += 1
    env.close()
    return ep_length


@pytest.fixture(scope="function")
def config(tmp_path):
    total_episode = 5
    data_save_path = tmp_path / "data.pkl"
    gen_data(total_episode, data_save_path)

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--expert_data", str(data_save_path)])
    return cfg


@pytest.fixture(scope="function", params=[True, False])
def asynchronous(request):
    return request.param


@pytest.mark.unittest
def test_offline_env(asynchronous, config):
    # create environment
    env = make("OfflineEnv", env_num=2, cfg=config, asynchronous=asynchronous)

    for ep_index in range(10):
        done = False
        step = 0
        env.reset()

        while not np.all(done):
            obs, reward, done, info = env.step(env.random_action())
            step += 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
