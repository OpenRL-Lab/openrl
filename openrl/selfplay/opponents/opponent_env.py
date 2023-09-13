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
from typing import Optional, Union

from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType, spaces

from openrl.envs.wrappers.base_wrapper import BaseWrapper


class BaseOpponentEnv:
    def __init__(self, env, opponent_player: str):
        self.env = env
        self.opponent_player = opponent_player
        self._action_space: Optional[spaces.Space[WrapperActType]] = None
        self._observation_space: Optional[spaces.Space[WrapperObsType]] = None

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        if self._action_space is None:
            action_space = self.env.action_space(self.opponent_player)
            if isinstance(action_space, list):
                action_space = action_space[0]
            return action_space
        return self.env.action_space

    @action_space.setter
    def action_space(self, space: spaces.Space[WrapperActType]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        if self._observation_space is None:
            return self.env.observation_space(self.opponent_player)
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space[WrapperObsType]):
        self._observation_space = space

    @property
    def agent_num(self) -> int:
        if isinstance(self.env, BaseWrapper) and hasattr(self.env, "agent_num"):
            return self.env.agent_num
        else:
            return self._agent_num()

    @property
    def parallel_env_num(self) -> int:
        return 1

    def _agent_num(self) -> int:
        return 1

    def process_obs(self, observation, termination, truncation, info):
        return observation, termination, truncation, info

    def process_action(self, action):
        return action

    def reset(self, **kwargs):
        pass
