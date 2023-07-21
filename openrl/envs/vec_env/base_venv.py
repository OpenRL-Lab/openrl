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
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, Union

import gymnasium as gym
import numpy as np

from openrl.envs.vec_env.utils.numpy_utils import single_random_action
from openrl.envs.vec_env.utils.util import prepare_action_masks, tile_images
from openrl.envs.wrappers.base_wrapper import BaseWrapper
from openrl.envs.wrappers.util import is_wrapped

IN_COLAB = "google.colab" in sys.modules

# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]


class BaseVecEnv(
    ABC,
):
    """
    An abstract vectorized environment.

    :param parallel_env_num: Number of environments
    :param observation_space: Observation space
    :param action_space: Action space
    """

    metadata = {
        "render.modes": [
            "human",
            "rgb_array",
            "group_human",
            "group_rgb_array",
            "single_human",
            "single_rgb_array",
        ]
    }

    observation_space: gym.Space
    action_space: gym.Space

    parallel_env_num: int

    closed = False

    _np_random: Optional[np.random.Generator] = None

    def __init__(
        self,
        parallel_env_num: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        render_mode: Optional[str] = None,
        auto_reset: bool = True,
    ):
        self.parallel_env_num = parallel_env_num
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode
        self.closed = False
        self.viewer = None
        self.auto_reset = auto_reset

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_send is still doing work, that work will
        be cancelled and step_fetch() should not be called
        until step_send() is invoked again.

        :return: observation
        """
        results = self._reset(seed=seed, options=options)
        self.vector_render()
        return results

    def _reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        raise NotImplementedError

    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        results = self._step(actions)
        self.vector_render()
        return results

    def _step(self, actions):
        raise NotImplementedError

    def vector_render(self):
        if self.render_mode is None or self.render_mode in [
            "rgb_array",
            "human",
            "single_rgb_array",
        ]:
            return

        if self.render_mode == "group_human":
            self.render("human")
        elif self.render_mode == "group_rgb_array":
            pass
        else:
            raise NotImplementedError(
                "render_mode {} is not implemented".format(self.render_mode)
            )

    def close(self, **kwargs) -> None:
        """
        Clean up the environment's resources.
        """
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def close_extras(self, **kwargs):
        """Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering

        :param mode: the rendering type
        """
        try:
            imgs = self._get_images()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == "human":
            if IN_COLAB:
                from google.colab.patches import cv2_imshow

                cv2_imshow(bigimg[:, :, ::-1])
            else:
                import cv2  # pytype:disable=import-error

                cv2.imshow("Vec_Env:{}".format(self.env_name), bigimg[:, :, ::-1])
                cv2.waitKey(1)
        elif mode in [None, "rgb_array"]:
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

    def _get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Environment seeds can be passed to this reset argument in the future.
        The old ``.seed()`` method is being deprecated.
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    def __del__(self):
        """Closes the vector environment."""
        if not getattr(self, "closed", True):
            self.close()

    @property
    def unwrapped(self) -> "BaseVecEnv":
        return self

    @property
    @abstractmethod
    def env_name(self):
        return None

    @property
    def agent_num(self):
        return self._agent_num

    def call_send(self, name, *args, **kwargs):
        """Calls a method name for each parallel environment asynchronously."""

    def call_fetch(self, **kwargs) -> List[Any]:  # type: ignore
        """After calling a method in :meth:`call_send`, this function collects the results."""

    def call(self, name: str, *args, **kwargs) -> List[Any]:
        """Call a method, or get a property, from each parallel environment.

        Args:
            name (str): Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_send(name, *args, **kwargs)
        return self.call_fetch()

    def exec_func_send(self, func: Callable, indices, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            func: a function.
            indices: Indices of the environments to call the method on.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_send` while waiting for a pending call to complete
        """

    def exec_func_fetch(self, timeout: Union[int, float, None] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_fetch` times out.
                If `None` (default), the call to `step_fetch` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_fetch` without any prior call to `call_send`.
            TimeoutError: The call to `call_fetch` has timed out after timeout second(s).
        """

    def exec_func(
        self, func: Callable, indices: List[int], *args, **kwargs
    ) -> List[Any]:
        """Call a method, or get a property, from each parallel environment.

        Args:
            func : Name of the method to call.
            indices: Indices of the environments to call the method on.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.exec_func_send(func, indices, *args, **kwargs)
        return self.exec_func_fetch()

    def get_attr(self, name: str):
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Set a property in each sub-environment.

        Args:
            name (str): Name of the property to be set in each individual environment.
            values (list, tuple, or object): Values of the property to be set to. If `values` is a list or
                tuple, then it corresponds to the values for each individual environment, otherwise a single value
                is set for all environments.
        """

    def random_action(self, infos: Optional[List[Dict[str, Any]]] = None):
        """
        Get a random action from the action space
        """
        action_masks = prepare_action_masks(
            infos, agent_num=self.agent_num, as_batch=False
        )
        print(action_masks)
        return np.array(
            [
                [
                    single_random_action(
                        self.action_space,
                        (
                            action_masks[env_index][agent_index]
                            if action_masks is not None
                            else None
                        ),
                    )
                    for agent_index in range(self.agent_num)
                ]
                for env_index in range(self.parallel_env_num)
            ]
        )

    def env_is_wrapped(
        self, wrapper_class: Type[BaseWrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        indices = self._get_indices(indices)
        results = self.exec_func(
            is_wrapped, indices=indices, wrapper_class=wrapper_class
        )
        return [results[i] for i in indices]

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.parallel_env_num)
        elif isinstance(indices, int):
            indices = [indices]
        return indices
