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
import copy

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.extra_wrappers import AddStep

env_wrappers = [AddStep]


def train(Agent, Net, env_name, env_num, total_time_steps):
    # create environment, set environment parallelism to 9
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "train.yaml"])
    env = make(env_name, env_num=env_num, cfg=cfg, env_wrappers=env_wrappers)
    # create the neural network

    net = Net(
        env,
        cfg=cfg,
    )
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=total_time_steps)
    env.close()
    return agent


def evaluation(agent, env_name):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    env = make(env_name, env_num=2, env_wrappers=env_wrappers, asynchronous=False)
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()

    done = False
    step = 0
    total_reward = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        pre_obs = copy.copy(obs)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)

        print(
            f"{step}: obs:{pre_obs[..., 0].flatten()}, action: {action.flatten()},"
            f" ,reward:{np.mean(r)}"
            # f"{step}: obs:{pre_obs}, action: {action.flatten()}, ,reward:{np.mean(r)}"
        )
    env.close()
    print("total reward:", total_reward)


def test_env():
    env = make("IdentityEnv", env_num=1, asynchronous=False)
    obs, info = env.reset()
    print(obs)
    done = False
    step = 0
    while not np.any(done):
        action = env.random_action()
        obs, r, done, info = env.step(action)
        step += 1

    env.close()
