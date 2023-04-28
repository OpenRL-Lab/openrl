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
from typing import Any, Dict, Optional

from gymnasium.core import ActType

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.vec_info.base_vec_info import BaseVecInfo
from openrl.envs.vec_env.wrappers.base_wrapper import VecEnvWrapper


class VecMonitorWrapper(VecEnvWrapper):
    def __init__(self, vec_info: BaseVecInfo, env: BaseVecEnv):
        super().__init__(env)
        self.vec_info = vec_info

    @property
    def use_monitor(self):
        return True

    def step(self, action: ActType, extra_data: Optional[Dict[str, Any]] = None):
        returns = self.env.step(action, extra_data)
        self.vec_info.append(info=returns[-1])

        return returns

    def statistics(self, buffer):  # TODO
        # this function should be called each episode
        info_dict = self.vec_info.statistics(buffer)
        self.vec_info.reset()
        return info_dict
