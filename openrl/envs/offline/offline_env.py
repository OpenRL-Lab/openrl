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
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding

from openrl.datasets.expert_dataset import ExpertDataset

class OfflineEnv(gym.Env):
    _np_random: Optional[np.random.Generator] = None

    def __init__(self,dataset_path):
        self.dataset = ExpertDataset(dataset_path)
        self.observation_space = self.dataset.observation_space
        self.action_space = self.dataset.action_space
        self.agent_num = self.dataset.agent_num

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        return None, {}

    def step(self, action):
        return None, 0.0, False, {}


    def close(self):
        pass