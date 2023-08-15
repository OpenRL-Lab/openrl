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
from openrl.envs.snake.snake import SnakeEatBeans

env = SnakeEatBeans()

obs, info = env.reset()

done = False
while not np.any(done):
    a1 = np.zeros(4)
    a1[env.action_space.sample()] = 1
    a2 = np.zeros(4)
    a2[env.action_space.sample()] = 1
    obs, reward, done, info = env.step([a1, a2])
    print("obs:", obs, reward, "\ndone:", done, info)
