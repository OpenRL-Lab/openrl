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
import copy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from openrl.selfplay.wrappers.base_multiplayer_wrapper import BaseMultiPlayerWrapper


class RandomOpponentWrapper(BaseMultiPlayerWrapper):
    self_player: Optional[str] = None

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            if self.self_player is None:
                self.env.reset()
                self.self_player = self.np_random.choice(self.env.agents)
            return self.env.action_spaces[self.self_player]
        return self._action_space

    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            if self.self_player is None:
                self.env.reset()
                self.self_player = self.np_random.choice(self.env.agents)
            return self.env.observation_spaces[self.self_player]
        return self._observation_space

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        while True:
            self.env.reset(seed=seed, **kwargs)
            self.self_player = self.np_random.choice(self.env.agents)

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                if termination or truncation:
                    assert False, "This should not happen"

                if self.self_player == agent:
                    return copy.copy(observation), info

                mask = observation["action_mask"]
                action = self.env.action_space(agent).sample(
                    mask
                )  # this is where you would insert your policy
                self.env.step(action)

    def step(self, action):
        self.env.step(action)
        while True:
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                if self.self_player == agent:
                    return copy.copy(observation), reward, termination, truncation, info

                if termination or truncation:
                    return (
                        copy.copy(self.env.observe(self.self_player)),
                        self.env.rewards[self.self_player],
                        termination,
                        truncation,
                        self.env.infos[self.self_player],
                    )

                else:
                    mask = observation["action_mask"]
                    action = self.env.action_space(agent).sample(
                        mask
                    )  # this is where you would insert your policy
                    self.env.step(action)
