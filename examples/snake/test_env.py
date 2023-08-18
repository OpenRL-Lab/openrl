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
from wrappers import ConvertObs

from openrl.envs.snake.snake import SnakeEatBeans
from openrl.envs.snake.snake_pettingzoo import SnakeEatBeansAECEnv
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


def test_raw_env():
    env = SnakeEatBeans()

    obs, info = env.reset()

    done = False
    while not np.any(done):
        a1 = np.zeros(4)
        a1[env.action_space.sample()] = 1
        a2 = np.zeros(4)
        a2[env.action_space.sample()] = 1
        obs, reward, done, info = env.step([a1, a2])
        print("obs:", obs)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)


def test_aec_env():
    from PIL import Image

    img_list = []
    env = SnakeEatBeansAECEnv(render_mode="rgb_array")
    env.reset(seed=0)
    # time.sleep(1)
    img = env.render()
    img_list.append(img)
    step = 0
    for player_name in env.agent_iter():
        if step > 20:
            break
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        action = env.action_space(player_name).sample()
        # if player_name == "player_0":
        #     action = 2
        # elif player_name == "player_1":
        #     action = 3
        # else:
        #     raise ValueError("Unknown player name: {}".format(player_name))
        env.step(action)
        img = env.render()
        if player_name == "player_0":
            img_list.append(img)
        # time.sleep(1)

        step += 1
    print("Total steps: {}".format(step))

    save_path = "test.gif"
    img_list = [Image.fromarray(img) for img in img_list]
    img_list[0].save(save_path, save_all=True, append_images=img_list[1:], duration=500)


def test_vec_env():
    from openrl.envs.common import make

    env = make(
        "snakes_1v1",
        opponent_wrappers=[
            RandomOpponentWrapper,
        ],
        env_wrappers=[ConvertObs],
        render_mode="group_human",
        env_num=2,
    )
    obs, info = env.reset()
    step = 0
    done = False
    while not np.any(done):
        action = env.random_action()
        obs, reward, done, info = env.step(action)
        time.sleep(0.3)
        step += 1
    print("Total steps: {}".format(step))


if __name__ == "__main__":
    test_vec_env()
