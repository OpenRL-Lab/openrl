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
from openrl.envs.vec_env.wrappers.base_wrapper import VectorObservationWrapper


class MATWrapper(VectorObservationWrapper):
    @property
    def observation_space(
        self,
    ):
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            observation_space = self.env.observation_space
        else:
            observation_space = self._observation_space

        if (
            "critic" in observation_space.spaces.keys()
            and "policy" in observation_space.spaces.keys()
        ):
            observation_space = observation_space["policy"]
        return observation_space

    def observation(self, observation):
        if self._observation_space is None:
            observation_space = self.env.observation_space
        else:
            observation_space = self._observation_space

        if (
            "critic" in observation_space.spaces.keys()
            and "policy" in observation_space.spaces.keys()
        ):
            observation = observation["policy"]
        return observation
