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
from openrl.envs.wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # create environment
    env = make("SuperMarioBros-1-1-v1", env_num=2)
    # create the neural network
    net = Net(env, device="cuda")
    # initialize the trainer
    agent = Agent(net)
    # start training
    agent.train(total_time_steps=2000)
    # save the trained model
    agent.save("super_mario_agent/")
    # close the environment
    env.close()
    return agent


def game_test():
    # begin to test
    env = make(
        "SuperMarioBros-1-1-v1",
        render_mode="group_human",
        env_num=1,
    )

    # Save the running result as a GIF image.
    env = GIFWrapper(env, "super_mario.gif")

    # init the agent
    agent = Agent(Net(env))
    # set up the environment and initialize the RNN network.
    agent.set_env(env)
    # load the trained model
    agent.load("super_mario_agent/")

    # 开始测试
    obs, info = env.reset()
    step = 0
    while True:
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        print(f"{step}: reward:{np.mean(r)}")

        if any(done):
            break

    env.close()


if __name__ == "__main__":
    agent = train()
    game_test()
