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
    # 创建环境
    env = make("SuperMarioBros-1-1-v1", env_num=2)
    # 创建网络
    net = Net(env, device="cuda")
    # 初始化训练器
    agent = Agent(net)
    # 开始训练
    agent.train(total_time_steps=2000)
    # 保存模型
    agent.save("super_mario_agent/")
    # 关闭环境
    env.close()
    return agent


def game_test():
    # 开始测试环境
    env = make(
        "SuperMarioBros-1-1-v1",
        render_mode="group_human",
        env_num=1,
    )

    # 保存运行结果为GIF图片
    env = GIFWrapper(env, "super_mario.gif")

    # 初始化网络
    agent = Agent(Net(env))
    # 设置环境，并初始化RNN网络
    agent.set_env(env)
    # 加载模型
    agent.load("super_mario_agent/")

    # 开始测试
    obs, info = env.reset()
    step = 0
    while True:
        # 智能体根据 observation 预测下一个动作
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
