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

from typing import Any, Optional, Type

import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box

from openrl.envs.wrappers.base_wrapper import BaseWrapper


def nest_expand_dim(input: Any) -> Any:
    if isinstance(input, (np.ndarray, float, int)):
        return np.expand_dims(input, 0)
    elif isinstance(input, list):
        return [input]
    elif isinstance(input, dict):
        for key in input:
            input[key] = nest_expand_dim(input[key])
        return input
    elif isinstance(input, Box):
        return [input]
    elif isinstance(input, np.int64):
        return [input]
    else:
        raise NotImplementedError("Not support type: {}".format(type(input)))


def unwrap_wrapper(
    env: gym.Env, wrapper_class: Type[BaseWrapper]
) -> Optional[BaseWrapper]:
    """
    Retrieve a ``BaseWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, BaseWrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: gym.Env, wrapper_class: Type[BaseWrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None
