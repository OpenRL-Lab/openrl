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
from typing import Any, Dict, Optional, SupportsFloat, Tuple, TypeVar, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType

ArrayType = TypeVar("ArrayType")


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

    def set_render_mode(self, render_mode: Union[None, str]):
        if hasattr(self.env, "set_render_mode"):
            self.env.set_render_mode(render_mode)


class BaseObservationWrapper(BaseWrapper):
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        results = self.env.step(action)
        observation = results[0]
        new_obs = self.observation(observation)
        return new_obs, *results[1:]

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError


class BaseRewardWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        returns = self.env.step(action)
        return returns[0], self.reward(returns[1]), *returns[2:]

    def reward(self, reward: ArrayType) -> ArrayType:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError
