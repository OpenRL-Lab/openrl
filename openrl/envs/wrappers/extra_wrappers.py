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
from copy import deepcopy

import gymnasium as gym
from gymnasium.wrappers import AutoResetWrapper, StepAPICompatibility

from openrl.envs.wrappers import BaseObservationWrapper, BaseWrapper


class RemoveTruncated(StepAPICompatibility, BaseWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        output_truncation_bool = False
        super().__init__(env, output_truncation_bool=output_truncation_bool)


class AutoReset(AutoResetWrapper, BaseWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)

    @property
    def has_auto_reset(self):
        return True


class DictWrapper(BaseObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        need_convert = "Dict" not in self.env.observation_space.__class__.__name__
        if need_convert:
            self.observation_space = gym.spaces.Dict(
                {
                    "policy": self.env.observation_space,
                    "critic": self.env.observation_space,
                }
            )

    def observation(self, observation):
        return {"policy": observation, "critic": deepcopy(observation)}


class GIFWrapper(BaseWrapper):
    def __init__(self, env, gif_path: str):
        super().__init__(env)
        self.gif_path = gif_path
        import imageio

        self.writter = imageio.get_writer(self.gif_path, mode="I")

    def reset(self, **kwargs):
        results = self.env.reset(**kwargs)
        img = self.env.render()
        self.writter.append_data(img)
        return results

    def step(self, action):
        results = self.env.step(action)
        img = self.env.render()
        self.writter.append_data(img)
        return results

    def close(self):
        self.env.close()
        self.writter.close()
