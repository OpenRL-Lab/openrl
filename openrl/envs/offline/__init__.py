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

from gymnasium import Env

from openrl.envs.common import build_envs
from openrl.envs.offline.offline_env import OfflineEnv


def offline_make(dataset, render_mode, disable_env_checker, **kwargs):
    env_id = kwargs["env_id"]
    env_num = kwargs["env_num"]
    seed = kwargs.pop("seed", None)
    assert seed is not None, "seed must be set"

    env = OfflineEnv(dataset, env_id, env_num, seed)
    return env


def make_offline_envs(
    dataset: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_wrappers += []

    env_fns = build_envs(
        make=offline_make,
        id=dataset,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        need_env_id=True,
        **kwargs,
    )
    return env_fns
