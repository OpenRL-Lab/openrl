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
import gymnasium as gym


class BaseWrapper(gym.Wrapper):
    def __init__(self, env, reward_class=None) -> None:
        super().__init__(env)
        self.reward_class = reward_class

    def step(self, action):
        return super().step(action)

    @property
    def env_name(self):
        if hasattr(self.env, "env_name"):
            return self.env.env_name
        return self.env.unwrapped.spec.id

    @property
    def agent_num(self):
        if hasattr(self.env, "agent_num"):
            return self.env.agent_num
        else:
            raise NotImplementedError("Not support agent_num")

    @property
    def use_monitor(self):
        return False

    @property
    def has_auto_reset(self):
        if hasattr(self.env, "has_auto_reset"):
            return self.env.has_auto_reset
        else:
            return False


class BaseObservationWrapper(BaseWrapper, gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
