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
import time

import numpy as np
import gym_pybullet_drones
import gymnasium as gym

from openrl.envs.common import make
def test_env():
    env = gym.make("hover-aviary-v0",gui=False,record=False)
    print("obs space:",env.observation_space)
    print("action space:",env.action_space)
    obs, info = env.reset(seed=42, options={})
    totoal_step  =0
    totol_reward = 0.
    while True:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        totoal_step+=1
        totol_reward+=reward
        # env.render()
        # time.sleep(1)
        if done:
            break
    print("total step:",totoal_step)
    print("total reward:",totol_reward)

def test_vec_env():
    env = make("pybullet_drones/hover-aviary-v0",env_num=2,gui=False,record=False,asynchronous=True)
    info,obs = env.reset(seed=0)
    totoal_step = 0
    totol_reward = 0.
    while True:
        obs, reward, done, info =  env.step(env.random_action())
        totoal_step+=1
        totol_reward+=np.mean(reward)
        if np.any(done) or totoal_step>100:
            break
    env.close()
    print("total step:", totoal_step)
    print("total reward:", totol_reward)




if __name__ == '__main__':
    test_env()
    # test_vec_env()