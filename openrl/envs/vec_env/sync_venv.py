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
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
from gymnasium import Env
from gymnasium.core import ActType
from gymnasium.spaces import Space

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.utils.numpy_utils import (
    concatenate,
    create_empty_array,
    iterate_action,
)


class SyncVectorEnv(BaseVecEnv):
    """Vectorized environment that serially runs multiple environments."""

    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
        render_mode: Optional[str] = None,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]

        self.copy = copy
        self.metadata = self.envs[0].metadata
        self._subenv_auto_reset = (
            hasattr(self.envs[0], "has_auto_reset") and self.envs[0].has_auto_reset
        )

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            parallel_env_num=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
            render_mode=render_mode,
        )

        self._check_spaces()
        self._agent_num = self.envs[0].agent_num

        self.observations = create_empty_array(
            self.observation_space,
            n=self.parallel_env_num,
            agent_num=self._agent_num,
            fn=np.zeros,
        )

        self._rewards = np.zeros(
            (self.parallel_env_num, self._agent_num, 1), dtype=np.float64
        )
        self._terminateds = np.zeros(
            (
                self.parallel_env_num,
                self._agent_num,
            ),
            dtype=np.bool_,
        )
        self._truncateds = np.zeros(
            (
                self.parallel_env_num,
                self._agent_num,
            ),
            dtype=np.bool_,
        )
        self._actions = None

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.parallel_env_num)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.parallel_env_num)]
        assert len(seed) == self.parallel_env_num

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def _reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.parallel_env_num)]
        if isinstance(seed, int):
            seed = [seed + i * 10086 for i in range(self.parallel_env_num)]
        assert len(seed) == self.parallel_env_num

        self._terminateds[:] = False
        self._truncateds[:] = False

        observations = []
        infos = []

        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            returns = env.reset(**kwargs)

            if isinstance(returns, tuple):
                if len(returns) == 2:
                    # obs, info
                    observations.append(returns[0])
                    infos.append(returns[1])
                else:
                    raise NotImplementedError(
                        "Not support reset return length: {}".format(len(returns))
                    )
            else:
                observations.append(returns)

        if len(infos) > 0:
            return self.format_obs(observations), infos
        else:
            return self.format_obs(observations)

    def format_obs(self, observations: Iterable) -> Union[tuple, dict, np.ndarray]:
        self.observations = concatenate(
            self.observation_space, observations, self.observations
        )
        return deepcopy(self.observations) if self.copy else self.observations

    def _step(self, actions: ActType):
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        _actions = iterate_action(self.action_space, actions)

        observations, infos = [], []

        for i, (env, action) in enumerate(zip(self.envs, _actions)):
            returns = env.step(action)
            assert isinstance(
                returns, tuple
            ), "step return must be tuple, but got: {}".format(type(returns))

            _need_reset = not self._subenv_auto_reset
            if len(returns) == 5:
                (
                    observation,
                    self._rewards[i],
                    self._terminateds[i],
                    self._truncateds[i],
                    info,
                ) = returns
                need_reset = _need_reset and (
                    all(self._terminateds[i]) or all(self._truncateds[i])
                )
            else:
                (
                    observation,
                    self._rewards[i],
                    self._terminateds[i],
                    info,
                ) = returns
                need_reset = _need_reset and all(self._terminateds[i])

            if need_reset:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info

            observations.append(observation)
            infos.append(info)

        if len(returns) == 5:
            return (
                self.format_obs(observations),
                np.copy(self._rewards),
                np.copy(self._terminateds),
                np.copy(self._truncateds),
                infos,
            )
        elif len(returns) == 4:
            return (
                self.format_obs(observations),
                np.copy(self._rewards),
                np.copy(self._terminateds),
                infos,
            )
        else:
            raise NotImplementedError(
                "Not support step return length: {}".format(len(returns))
            )

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True

    def _get_images(self) -> Sequence[np.ndarray]:
        if self.render_mode == "single_rgb_array":
            return [self.envs[0].render()]
        else:
            return [env.render() for env in self.envs]

    @property
    def env_name(self):
        if hasattr(self.envs[0], "env_name"):
            return self.envs[0].env_name
        else:
            return self.envs[0].unwrapped.spec.id

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.parallel_env_num)]
        if len(values) != self.parallel_env_num:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.parallel_env_num} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)
