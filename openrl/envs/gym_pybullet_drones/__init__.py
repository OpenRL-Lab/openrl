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
from typing import Callable, List, Optional, Union

import gymnasium as gym
from gymnasium import Env

from openrl.envs.common import build_envs


def make_single_agent_drone_env(id: str, render_mode, disable_env_checker, **kwargs):
    import gym_pybullet_drones

    prefix = "pybullet_drones/"
    assert id.startswith(prefix), "id must start with pybullet_drones/"
    kwargs.pop("cfg")

    env = gym.envs.registration.make(id[len(prefix) :], **kwargs)
    return env


def make_single_agent_drone_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    from openrl.envs.wrappers import (  # AutoReset,; DictWrapper,
        RemoveTruncated,
        Single2MultiAgentWrapper,
    )

    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_wrappers += [
        Single2MultiAgentWrapper,
        RemoveTruncated,
    ]

    env_fns = build_envs(
        make=make_single_agent_drone_env,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
