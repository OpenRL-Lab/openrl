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
from typing import Any

from openrl.envs.toy_envs.bit_flipping_env import BitFlippingEnv
from openrl.envs.toy_envs.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvcontinuous,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)
from openrl.envs.toy_envs.multi_input_envs import SimpleMultiObsEnv

__all__ = [
    "BitFlippingEnv",
    "FakeImageEnv",
    "IdentityEnv",
    "IdentityEnvcontinuous",
    "IdentityEnvBox",
    "IdentityEnvMultiBinary",
    "IdentityEnvMultiDiscrete",
    "SimpleMultiObsEnv",
]


import copy
from typing import Callable, List, Optional, Union

from gymnasium import Env

from openrl.envs.common import build_envs

env_dict = {
    "BitFlippingEnv": BitFlippingEnv,
    "FakeImageEnv": FakeImageEnv,
    "IdentityEnv": IdentityEnv,
    "IdentityEnvcontinuous": IdentityEnvcontinuous,
    "IdentityEnvBox": IdentityEnvBox,
    "IdentityEnvMultiBinary": IdentityEnvMultiBinary,
    "IdentityEnvMultiDiscrete": IdentityEnvMultiDiscrete,
    "SimpleMultiObsEnv": SimpleMultiObsEnv,
}


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Env:
    env = env_dict[id]()

    return env


def make_toy_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    from openrl.envs.wrappers import Single2MultiAgentWrapper

    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_wrappers += [
        Single2MultiAgentWrapper,
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
