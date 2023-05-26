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

from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # create environment, set environment parallelism to 9
    env = make("InvertedPendulum-v4", env_num=9)
    # create the neural network
    net = Net(env)
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=30000)
    env.close()
    return agent


def evaluation(agent):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    env = make("InvertedPendulum-v4", render_mode=None, env_num=9, asynchronous=False)

    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()

    done = False
    step = 0
    totoal_reward = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 100 == 0:
            print(f"{step}: reward:{np.mean(r)}")
        totoal_reward += np.mean(r)
    env.close()
    print(f"total reward: {totoal_reward}")


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
