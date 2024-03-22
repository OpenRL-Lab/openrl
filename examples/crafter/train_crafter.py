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

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # create environment
    env = make("Crafter", env_num=32, asynchronous=True)
    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    # create the neural network
    net = Net(env, cfg=cfg, device="cuda")
    # initialize the trainer
    agent = Agent(net, use_wandb=True)
    # start training
    agent.train(total_time_steps=100000000)
    # save the trained model
    agent.save("crafter_agent-100M/")
    # close the environment
    env.close()
    return agent


if __name__ == "__main__":
    agent = train()
