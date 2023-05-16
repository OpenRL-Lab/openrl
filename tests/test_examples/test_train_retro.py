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
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


@pytest.fixture(scope="module", params=[""])
def config(request):
    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_train_retro(config):
    env = make("Airstriker-Genesis", state="Level1", env_num=2, asynchronous=True)

    agent = Agent(Net(env, cfg=config))
    agent.train(total_time_steps=1000)

    agent.set_env(env)
    obs, info = env.reset()
    done = False
    total_reward = 0
    step = 0
    while True:
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        total_reward += np.mean(r)

        step += 1
        if step > 200 or any(done):
            break

    assert total_reward <= 300, "Retro Environment should be solved."

    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
