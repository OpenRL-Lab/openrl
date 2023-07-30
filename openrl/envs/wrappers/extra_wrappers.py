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
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import AutoResetWrapper, StepAPICompatibility

from openrl.envs.wrappers import BaseObservationWrapper, BaseRewardWrapper, BaseWrapper
from openrl.envs.wrappers.base_wrapper import ActType, ArrayType, WrapperObsType
from openrl.envs.wrappers.flatten import flatten


class RemoveTruncated(StepAPICompatibility, BaseWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        output_truncation_bool = False
        super().__init__(env, output_truncation_bool=output_truncation_bool)


class FlattenObservation(BaseObservationWrapper):
    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        BaseObservationWrapper.__init__(self, env)

        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """

        return flatten(self.env.observation_space, self.agent_num, observation)


class AddStep(BaseObservationWrapper):
    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """

        BaseObservationWrapper.__init__(self, env)
        assert isinstance(self.env.observation_space, spaces.Box) or isinstance(
            self.env.observation_space, spaces.Discrete
        )
        if isinstance(self.env.observation_space, spaces.Box):
            assert len(self.env.observation_space.shape) == 1

            self.observation_space = spaces.Box(
                np.append(self.env.observation_space.low, 0),
                np.append(self.env.observation_space.high, np.inf),
                shape=(self.env.observation_space.shape[0] + 1,),
            )
        else:
            self.observation_space = spaces.Discrete(n=self.env.observation_space.n + 1)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        self.step_count = 0
        return super().reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        return super().step(action)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        new_obs = np.append(observation, self.step_count)
        return new_obs


class MoveActionMask2InfoWrapper(BaseWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        self.need_convert = False
        if "action_mask" in self.env.observation_space.spaces.keys():
            self.need_convert = True
            self.observation_space = self.env.observation_space.spaces["observation"]

    def step(self, action):
        results = self.env.step(action)

        if self.need_convert:
            obs = results[0]["observation"]
            info = results[-1]
            info["action_masks"] = results[0]["action_mask"]
            return obs, *results[1:-1], info
        if "action_mask" in results[-1]:
            info = results[-1]
            info["action_masks"] = info["action_mask"]
            del info["action_mask"]
            return *results[0:-1], info
        return results

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.need_convert:
            info["action_masks"] = obs["action_mask"]
            obs = obs["observation"]
        else:
            if "action_mask" in info:
                info["action_masks"] = info["action_mask"]
                del info["action_mask"]
        return obs, info


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


class RecordReward(BaseWrapper):
    @property
    def has_auto_reset(self):
        return True


class ZeroRewardWrapper(BaseRewardWrapper):
    def reward(self, reward: ArrayType) -> ArrayType:
        return np.zeros_like(reward)
