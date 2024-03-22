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

from openrl.configs.config import create_config_parser

def render():
    # begin to test
    env = make(
        "Crafter",
        render_mode="human",
        env_num=1,
    )
    
    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # init the agent
    agent = Agent(Net(env, cfg=cfg))
    # set up the environment and initialize the RNN network.
    agent.set_env(env)
    # load the trained model
    agent.load("crafter_agent/")

    # begin to test
    trajectory = []
    obs, info = env.reset()
    step = 0
    while True:
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1

        if all(done):
            break
        
        img = obs["policy"][0,0]
        img = img.transpose((1, 2, 0))
        trajectory.append(img)
        
    # save the trajectory to gif
    import imageio
    imageio.mimsave("run_results/crafter.gif", trajectory, duration=0.01)

    env.close()

if __name__ == "__main__":
    render()
