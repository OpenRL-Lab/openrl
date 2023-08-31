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
from typing import List, Optional, Union

from openrl.envs.common import build_envs
from openrl.envs.snake.snake_pettingzoo import SnakeEatBeansAECEnv
from openrl.envs.wrappers.pettingzoo_wrappers import (
    OpenRLAssertOutOfBoundsWrapper,
    OpenRLOrderEnforcingWrapper,
    SeedEnv,
)


def snake_env_make(id: str, render_mode, disable_env_checker, **kwargs):
    if id.startswith("snakes"):
        env = SnakeEatBeansAECEnv(render_mode=render_mode, id=id)
    else:
        raise ValueError("Unknown env {}".format(id))
    return env


def make_snake_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    from openrl.envs.wrappers import RemoveTruncated, Single2MultiAgentWrapper

    env_wrappers = [
        OpenRLAssertOutOfBoundsWrapper,
        OpenRLOrderEnforcingWrapper,
        SeedEnv,
    ]
    env_wrappers += copy.copy(kwargs.pop("opponent_wrappers", []))
    env_wrappers += [
        RemoveTruncated,
    ]
    env_wrappers += copy.copy(kwargs.pop("env_wrappers", []))

    env_fns = build_envs(
        make=snake_env_make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )

    return env_fns
