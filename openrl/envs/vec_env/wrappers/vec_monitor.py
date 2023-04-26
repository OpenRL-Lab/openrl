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
from openrl.envs.vec_env.wrappers.base_wrapper import VecEnvWrapper
from openrl.envs.vec_env.wrappers.vec_info import NLPVecInfo, VecInfo

registed_vec_info = {
    "default": VecInfo,
    "NLPVecInfo": NLPVecInfo,
}


class VecInfoFactory:
    @staticmethod
    def get_vec_info_class(vec_info_class, env):
        if vec_info_class is None or vec_info_class.id is None:
            return registed_vec_info["default"](env.parallel_env_num, env.agent_num)
        return registed_vec_info[vec_info_class.id](
            env.parallel_env_num, env.agent_num, **vec_info_class.args
        )

    @staticmethod
    def register(name, vec_info):
        registed_vec_info[name] = vec_info


class VecMonitor(VecEnvWrapper):
    def __init__(self, vec_info: Any, env: BaseVecEnv):
        super().__init__(env)
        self.vec_info = vec_info

    @property
    def use_monitor(self):
        return True

    def step(self, action: ActType, extra_data: Optional[Dict[str, Any]] = None):
        returns = self.env.step(action, extra_data)
        self.vec_info.append(reward=returns[1], info=returns[-1])

        return returns

    def statistics(self):
        # this function should be called each episode
        info_dict = self.vec_info.statistics()
        self.vec_info.reset()
        return info_dict
