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
from openrl.envs.wrappers.extra_wrappers import AddStep
from openrl.modules.common import SACNet as Net
from openrl.runners.common import SACAgent as Agent

env_wrappers = [AddStep]


@pytest.fixture(scope="module", params=[""])
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


def train(Agent, Net, env_name, env_num, total_time_steps, config):
    cfg = config
    env = make(env_name, env_num=env_num, cfg=cfg, env_wrappers=env_wrappers)

    net = Net(
        env,
        cfg=cfg,
    )
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=total_time_steps)
    env.close()


@pytest.mark.unittest
def test_sac_net(config):
    train(Agent, Net, "IdentityEnvcontinuous", 2, 100, config)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
