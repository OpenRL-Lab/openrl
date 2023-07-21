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

import gymnasium as gym

import openrl
from openrl.envs.vec_env import (
    AsyncVectorEnv,
    BaseVecEnv,
    RewardWrapper,
    SyncVectorEnv,
    VecMonitorWrapper,
)
from openrl.envs.vec_env.vec_info import VecInfoFactory
from openrl.rewards import RewardFactory


def make(
    id: str,
    cfg=None,
    env_num: int = 1,
    asynchronous: bool = False,
    add_monitor: bool = True,
    render_mode: Optional[str] = None,
    make_custom_envs: Optional[Callable] = None,
    auto_reset: bool = True,
    **kwargs,
) -> BaseVecEnv:
    if render_mode in [None, "human", "rgb_array"]:
        convert_render_mode = render_mode
    elif render_mode in ["group_human", "group_rgb_array"]:
        # will display all the envs (when render_mode == "group_human")
        # or return all the envs' images (when render_mode == "group_rgb_array")
        convert_render_mode = "rgb_array"
    elif render_mode == "single_human":
        # will only display the first env
        convert_render_mode = [None] * (env_num - 1)
        convert_render_mode = ["human"] + convert_render_mode
        render_mode = None
    elif render_mode == "single_rgb_array":
        # env.render() will only return the first env's image
        convert_render_mode = [None] * (env_num - 1)
        convert_render_mode = ["rgb_array"] + convert_render_mode
    else:
        raise NotImplementedError(f"render_mode {render_mode} is not supported.")

    if make_custom_envs is not None:
        env_fns = make_custom_envs(
            id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
        )
    else:
        if id in gym.envs.registry.keys():
            from openrl.envs.gymnasium import make_gym_envs

            env_fns = make_gym_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )
        elif id in openrl.envs.mpe_all_envs:
            from openrl.envs.mpe import make_mpe_envs

            env_fns = make_mpe_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )
        elif id in openrl.envs.nlp_all_envs:
            from openrl.envs.nlp import make_nlp_envs

            env_fns = make_nlp_envs(
                id=id,
                env_num=env_num,
                render_mode=convert_render_mode,
                cfg=cfg,
                **kwargs,
            )
        elif id in openrl.envs.toy_all_envs:
            from openrl.envs.toy_envs import make_toy_envs

            env_fns = make_toy_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )

        elif id[0:14] in openrl.envs.super_mario_all_envs:
            from openrl.envs.super_mario import make_super_mario_envs

            env_fns = make_super_mario_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )
        elif id in openrl.envs.connect_all_envs:
            from openrl.envs.connect_env import make_connect_envs

            env_fns = make_connect_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )
        elif id in openrl.envs.gridworld_all_envs:
            from openrl.envs.gridworld import make_gridworld_envs

            env_fns = make_gridworld_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )
        elif id in openrl.envs.offline_all_envs:
            from openrl.envs.offline import make_offline_envs

            assert cfg.expert_data is not None, (
                "expert_data must be provided for offline envs, you can set it in a"
                " YAML file or with `--expert_data dataset_path`"
            )
            kwargs["seed"] = cfg.seed
            env_fns = make_offline_envs(
                dataset=cfg.expert_data,
                env_num=env_num,
                render_mode=convert_render_mode,
                **kwargs,
            )
        elif id in openrl.envs.pettingzoo_all_envs:
            from openrl.envs.PettingZoo import make_PettingZoo_envs

            env_fns = make_PettingZoo_envs(
                id=id, env_num=env_num, render_mode=convert_render_mode, **kwargs
            )
        else:
            raise NotImplementedError(f"env {id} is not supported.")

    if asynchronous:
        env = AsyncVectorEnv(env_fns, render_mode=render_mode, auto_reset=auto_reset)
    else:
        env = SyncVectorEnv(env_fns, render_mode=render_mode, auto_reset=auto_reset)

    reward_class = cfg.reward_class if cfg else None
    reward_class = RewardFactory.get_reward_class(reward_class, env)

    env = RewardWrapper(env, reward_class)

    if add_monitor:
        vec_info_class = cfg.vec_info_class if cfg else None
        vec_info_class = VecInfoFactory.get_vec_info_class(vec_info_class, env)
        env = VecMonitorWrapper(vec_info_class, env)

    return env
