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
from typing import List, Optional, Union


def PettingZoo_make(id, render_mode, disable_env_checker, **kwargs):
    if id == "":
        from pettingzoo.classic import tictactoe_v3

        env = tictactoe_v3.env(render_mode=render_mode)
    else:
        raise NotImplementedError
    return env


def make_PettingZoo_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_fns = build_envs(
        make=PettingZoo_make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
