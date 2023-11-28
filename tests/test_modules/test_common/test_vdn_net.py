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

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.mat_wrapper import MATWrapper
from openrl.modules.common import VDNNet
from openrl.runners.common import VDNAgent as Agent


@pytest.fixture(scope="module", params=[""])
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_vdn_net(config):
    env_num = 2
    env = make(
        "simple_spread",
        env_num=env_num,
        asynchronous=True,
    )
    env = MATWrapper(env)

    net = VDNNet(env, cfg=config)
    # initialize the trainer
    agent = Agent(net)
    # start training
    agent.train(total_time_steps=100)
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
