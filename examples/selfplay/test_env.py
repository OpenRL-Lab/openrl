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
from tictactoe_render import TictactoeRender

from openrl.envs.common import make
from openrl.envs.wrappers import FlattenObservation
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


def test_env():
    env_num = 1
    render_model = None
    render_model = "human"
    env = make(
        "tictactoe_v3",
        render_mode=render_model,
        env_num=env_num,
        asynchronous=False,
        opponent_wrappers=[TictactoeRender, RandomOpponentWrapper],
        env_wrappers=[FlattenObservation],
    )

    obs, info = env.reset(seed=1)
    done = False
    step_num = 0
    while not done:
        action = env.random_action(info)

        obs, done, r, info = env.step(action)

        done = np.any(done)
        step_num += 1
        if done:
            print(
                "step:"
                f" {step_num},{[env_info['final_observation'] for env_info in info]}"
            )
        else:
            print(f"step: {step_num},{obs}")


if __name__ == "__main__":
    test_env()
