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


from openrl.envs.wrappers.base_wrapper import BaseWrapper
from openrl.envs.wrappers.util import nest_expand_dim


class Single2MultiAgentWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._agent_num = 1

    @property
    def agent_num(self):
        return self._agent_num

    def reset(self, *, seed=None, options=None):
        returns = self.env.reset(seed=seed, options=options)

        if isinstance(returns, tuple):
            if len(returns) == 2:
                # obs, info
                return nest_expand_dim(returns[0]), returns[1]
            else:
                raise NotImplementedError(
                    "Not support reset return length: {}".format(len(returns))
                )
        else:
            return nest_expand_dim(returns)

    def step(self, action):  # TODO: support for MultiDiscrete
        returns = self.env.step(action[0])

        assert isinstance(
            returns, tuple
        ), "step return must be tuple, but got: {}".format(type(returns))

        if isinstance(returns[1], (float, int)):
            reward = [returns[1]]
        else:
            reward = returns[1]

        if len(returns) == 4:
            # obs reward done info
            return (
                nest_expand_dim(returns[0]),
                nest_expand_dim(reward),
                nest_expand_dim(returns[2]),
                returns[3],
            )
        elif len(returns) == 5:
            # obs reward done truncated, info
            return (
                nest_expand_dim(returns[0]),
                nest_expand_dim(reward),
                nest_expand_dim(returns[2]),
                nest_expand_dim(returns[3]),
                returns[4],
            )
        else:
            raise NotImplementedError(
                "Not support step return length: {}".format(len(returns))
            )
