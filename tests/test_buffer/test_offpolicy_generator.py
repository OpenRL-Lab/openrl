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

from openrl.envs.common import make
from openrl.modules.common import DQNNet as Net
from openrl.runners.common import DQNAgent as Agent


@pytest.fixture(scope="module", params=["--episode_length 10"])
def episode_length(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        "--use_recurrent_policy false --use_joint_action_loss false",
    ],
)
def generator_type(request):
    return request.param


@pytest.fixture(scope="module", params=["--use_proper_time_limits false"])
def use_proper_time_limits(request):
    return request.param


@pytest.fixture(scope="module", params=["--use_popart false --use_valuenorm false"])
def use_popart(request):
    return request.param


@pytest.fixture(scope="module")
def config(use_proper_time_limits, use_popart, generator_type, episode_length):
    config_str = (
        use_proper_time_limits
        + " "
        + use_popart
        + " "
        + generator_type
        + " "
        + episode_length
    )

    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(config_str.split())
    return cfg


@pytest.mark.unittest
def test_buffer_generator(config):
    env = make("CartPole-v1", env_num=2)
    agent = Agent(Net(env, cfg=config))
    agent.train(total_time_steps=50)
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
