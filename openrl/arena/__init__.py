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
from typing import Callable, Optional

import openrl
from openrl.arena.two_player_arena import TwoPlayerArena
from openrl.envs import pettingzoo_all_envs


def make_arena(
    env_id: str,
    custom_build_env: Optional[Callable] = None,
    render: Optional[bool] = False,
    use_tqdm: Optional[bool] = True,
    **kwargs,
):
    if custom_build_env is None:
        if (
            env_id in pettingzoo_all_envs
            or env_id in openrl.envs.PettingZoo.registration.pettingzoo_env_dict.keys()
        ):
            from openrl.envs.PettingZoo import make_PettingZoo_env

            render_mode = None
            if render:
                render_mode = "human"
            env_fn = make_PettingZoo_env(env_id, render_mode=render_mode, **kwargs)
        else:
            raise ValueError(f"Unknown env_id: {env_id}")
    else:
        env_fn = custom_build_env(env_id, render, **kwargs)

    return TwoPlayerArena(env_fn, use_tqdm=use_tqdm)
