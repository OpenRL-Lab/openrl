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

from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.utils import seeding

from openrl.envs.vec_env.base_venv import BaseVecEnv, VecEnvIndices
from openrl.envs.wrappers import BaseWrapper

ArrayType = TypeVar("ArrayType")


class VecEnvWrapper(BaseVecEnv, ABC):
    """Wraps the vectorized environment to allow a modular transformation.

    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code.

    Note:
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, env: BaseVecEnv):
        assert isinstance(env, BaseVecEnv)
        self.env = env
        self._action_space: Optional[spaces.Space[WrapperActType]] = None
        self._observation_space: Optional[spaces.Space[WrapperObsType]] = None
        self._reward_range: Optional[Tuple[SupportsFloat, SupportsFloat]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._parallel_env_num = self.env.parallel_env_num

    @property
    def parallel_env_num(self) -> int:
        return self._parallel_env_num

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space[WrapperActType]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space[WrapperObsType]):
        self._observation_space = space

    @property
    def reward_range(self) -> Tuple[SupportsFloat, SupportsFloat]:
        """Return the :attr:`Env` :attr:`reward_range` unless overwritten then the wrapper :attr:`reward_range` is used."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[SupportsFloat, SupportsFloat]):
        self._reward_range = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the :attr:`Env` :attr:`metadata`."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def render_mode(self) -> Optional[str]:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    def random_action(self, infos=None):
        return self.env.random_action(infos=infos)

    def reset(self, **kwargs):
        """Reset all environments."""
        return self.env.reset(**kwargs)

    def step(self, actions, *args, **kwargs):
        """Step all environments."""
        return self.env.step(actions, *args, **kwargs)

    def close(self, **kwargs):
        return self.env.close(**kwargs)

    def close_extras(self, **kwargs):
        return self.env.close_extras(**kwargs)

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
    def unwrapped(self):
        return self.env.unwrapped

    def _get_images(self) -> Sequence[np.ndarray]:
        return self.env._get_images()

    @property
    def env_name(self):
        if hasattr(self.env, "env_name"):
            return self.env.env_name
        return self.env.unwrapped.spec.id

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    def set_attr(self, name, values):
        return self.env.set_attr(name, values)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def __del__(self):
        self.env.__del__()

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def env_is_wrapped(
        self, wrapper_class: Type[BaseWrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return self.env.env_is_wrapped(wrapper_class, indices=indices)


class VectorObservationWrapper(VecEnvWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the observation. Equivalent to :class:`gym.ObservationWrapper` for vectorized environments."""

    def reset(self, **kwargs):
        """Modifies the observation returned from the environment ``reset`` using the :meth:`observation`."""
        results = self.env.reset(**kwargs)
        if isinstance(results, tuple) and len(results) == 2:
            observation, info = results
            return self.observation(observation), info
        else:
            observation = results
            return self.observation(observation)

    def step(self, actions, *args, **kwargs):
        """Modifies the observation returned from the environment ``step`` using the :meth:`observation`."""
        results = self.env.step(actions, *args, **kwargs)

        if len(results) == 5:
            observation, reward, termination, truncation, info = results
            return (
                self.observation(observation),
                reward,
                termination,
                truncation,
                info,
            )
        elif len(results) == 4:
            observation, reward, done, info = results
            return (
                self.observation(observation),
                reward,
                done,
                info,
            )
        else:
            raise ValueError(
                "Invalid step return value, expected 4 or 5 values, got {} values"
                .format(len(results))
            )

    def observation(self, observation: ObsType) -> ObsType:
        """Defines the observation transformation.

        Args:
            observation (object): the observation from the environment

        Returns:
            observation (object): the transformed observation
        """
        raise NotImplementedError


class VectorActionWrapper(VecEnvWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the actions. Equivalent of :class:`~gym.ActionWrapper` for vectorized environments."""

    def step(self, actions: ActType, *args, **kwargs):
        """Steps through the environment using a modified action by :meth:`action`."""
        return self.env.step(self.action(actions), *args, **kwargs)

    def actions(self, actions: ActType) -> ActType:
        """Transform the actions before sending them to the environment.

        Args:
            actions (ActType): the actions to transform

        Returns:
            ActType: the transformed actions
        """
        raise NotImplementedError


class VectorRewardWrapper(VecEnvWrapper):
    """Wraps the vectorized environment to allow a modular transformation of the reward. Equivalent of :class:`~gym.RewardWrapper` for vectorized environments."""

    def step(self, actions, *args, **kwargs):
        """Steps through the environment returning a reward modified by :meth:`reward`."""
        results = self.env.step(actions, *args, **kwargs)
        reward = self.reward(results[1])
        return results[0], reward, *results[2:]

    def reward(self, reward: ArrayType) -> ArrayType:
        """Transform the reward before returning it.

        Args:
            reward (array): the reward to transform

        Returns:
            array: the transformed reward
        """
        raise NotImplementedError
