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
import numpy as np
from gymnasium.spaces import Dict


def get_policy_obs(raw_obs):
    if isinstance(raw_obs, dict):
        if "policy" in raw_obs:
            return get_obs(raw_obs, model_name="policy")
    return get_obs(raw_obs)


def get_critic_obs(raw_obs):
    if isinstance(raw_obs, dict):
        if "critic" in raw_obs:
            return get_obs(raw_obs, model_name="critic")
    return get_obs(raw_obs)


def get_obs(raw_obs, model_name=None):
    if model_name is not None:
        raw_obs = raw_obs[model_name]
    return raw_obs


def get_policy_obs_space(obs_space):
    if isinstance(obs_space, Dict):
        if "policy" in obs_space.spaces:
            return get_shape_from_obs_space_v2(obs_space, model_name="policy")
    return get_shape_from_obs_space_v2(obs_space)


def get_critic_obs_space(obs_space):
    if isinstance(obs_space, Dict):
        if "critic" in obs_space.spaces:
            return get_shape_from_obs_space_v2(obs_space, model_name="critic")
    return get_shape_from_obs_space_v2(obs_space)


def get_shape_from_obs_space_v2(obs_space, model_name=None):
    if model_name is not None:
        obs_space = obs_space[model_name]
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == "Dict":
        obs_shape = obs_space.spaces
    elif obs_space.__class__.__name__ == "Discrete":
        obs_shape = (obs_space.n,)
    else:
        raise NotImplementedError(
            "obs_space type {} not supported".format(obs_space.__class__.__name__)
        )
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _flatten_v3(T, N, agent_num, x):
    return x.reshape(T * N * agent_num, *x.shape[3:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _cast_v3(x):
    return x.transpose(1, 0, 2, 3).reshape(-1, *x.shape[2:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols
