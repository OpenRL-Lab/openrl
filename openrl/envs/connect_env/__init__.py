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
from typing import Any, Callable, List, Optional, Union

from gymnasium import Env

from openrl.envs.common import build_envs
from openrl.envs.connect_env.connect3_env import Connect3Env
from openrl.envs.connect_env.connect4_env import Connect4Env


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Env:
    # create Connect3 environment from id
    if id == "connect3":
        env = Connect3Env(env_name=id)
    elif id == "connect4":
        env = Connect4Env(env_name=id)
    else:
        raise NotImplementedError(f"Unsupported environment: {id}")

    return env


def make_connect_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    from openrl.envs.wrappers import RemoveTruncated, Single2MultiAgentWrapper

    env_wrappers = [
        Single2MultiAgentWrapper,
        RemoveTruncated,
    ]
    env_fns = build_envs(
        make=make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
