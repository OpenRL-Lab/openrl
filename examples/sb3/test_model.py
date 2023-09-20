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

# Use OpenRL to load stable-baselines's model for testing

import numpy as np
import torch

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common.ppo_net import PPONet as Net
from openrl.modules.networks.policy_value_network_sb3 import (
    PolicyValueNetworkSB3 as PolicyValueNetwork,
)
from openrl.runners.common import PPOAgent as Agent


def evaluation(local_trained_file_path=None):
    # begin to test

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])

    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = "group_human"
    render_mode = None
    env = make("CartPole-v1", render_mode=render_mode, env_num=9, asynchronous=True)
    model_dict = {"model": PolicyValueNetwork}
    net = Net(
        env,
        cfg=cfg,
        model_dict=model_dict,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # initialize the trainer
    agent = Agent(
        net,
    )
    if local_trained_file_path is not None:
        agent.load(local_trained_file_path)
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False

    total_step = 0
    total_reward = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        total_step += 1
        total_reward += np.mean(r)
        if total_step % 50 == 0:
            print(f"{total_step}: reward:{np.mean(r)}")
    env.close()
    print("total step:", total_step)
    print("total reward:", total_reward)


if __name__ == "__main__":
    evaluation()
