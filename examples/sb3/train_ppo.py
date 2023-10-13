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
import torch
from test_model import evaluation

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common.ppo_net import PPONet as Net
from openrl.modules.networks.policy_value_network_sb3 import (
    PolicyValueNetworkSB3 as PolicyValueNetwork,
)
from openrl.runners.common import PPOAgent as Agent


def train_agent():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])

    env = make("CartPole-v1", env_num=8, asynchronous=True)

    model_dict = {"model": PolicyValueNetwork}
    net = Net(
        env,
        cfg=cfg,
        model_dict=model_dict,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000

    agent.train(total_time_steps=100000)
    env.close()

    agent.save("./ppo_sb3_agent")


if __name__ == "__main__":
    train_agent()
    evaluation(local_trained_file_path="./ppo_sb3_agent")
