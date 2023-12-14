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
from gymnasium import spaces


@pytest.fixture
def obs_space():
    return spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)


@pytest.fixture
def act_space():
    return spaces.Discrete(2)


@pytest.fixture(
    scope="module", params=["--use_share_model false", "--use_share_model true"]
)
def config(request):
    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.fixture
def amp_config():
    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args("")
    return cfg


@pytest.fixture
def init_module(config, obs_space, act_space):
    from openrl.modules.ppo_module import PPOModule

    module = PPOModule(
        config,
        policy_input_space=obs_space,
        critic_input_space=obs_space,
        act_space=act_space,
        share_model=config.use_share_model,
    )
    return module


@pytest.fixture
def buffer_data(config, obs_space, act_space):
    from openrl.buffers.normal_buffer import NormalReplayBuffer

    buffer = NormalReplayBuffer(
        config,
        num_agents=1,
        obs_space=obs_space,
        act_space=act_space,
        data_client=None,
        episode_length=100,
    )
    return buffer.data


@pytest.mark.unittest
def test_a2c_algorithm(config, init_module, buffer_data):
    from openrl.algorithms.a2c import A2CAlgorithm

    a2c_algo = A2CAlgorithm(config, init_module)

    a2c_algo.train(buffer_data)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
