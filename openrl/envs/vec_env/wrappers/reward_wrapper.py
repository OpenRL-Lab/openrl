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
from openrl.rewards.base_reward import BaseReward


class RewardWrapper(VecEnvWrapper):
    def __init__(self, env: BaseVecEnv, reward_class: Optional[BaseReward] = None):
        super().__init__(env)
        self.reward_class = reward_class
        if self.reward_class and len(self.reward_class.inner_reward_fn) > 0:
            env.call("set_reward", **{"reward_fn": self.reward_class.inner_reward_fn})

    def step(self, action: ActType, extra_data: Optional[Dict[str, Any]] = None):
        obs, rewards, dones, infos = self.env.step(action)

        if self.reward_class is not None and extra_data is not None:
            extra_data.update({"action": action})
            extra_data.update({"obs": obs})
            extra_data.update({"reward": rewards})
            extra_data.update({"done": dones})
            extra_data.update({"info": infos})
            rewards, new_infos = self.reward_class.get_reward(data=extra_data)

            if new_infos is not None:
                num_envs = len(infos)
                for i in range(num_envs):
                    infos[i] = {**infos[i], **new_infos[i]}

        return obs, rewards, dones, infos

    def batch_rewards(self, buffer):
        infos = {}
        if self.reward_class is not None:
            infos = self.reward_class.batch_rewards(buffer)
        return infos
