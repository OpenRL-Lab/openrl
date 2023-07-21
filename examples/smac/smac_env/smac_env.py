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

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .StarCraft2_Env import StarCraft2Env


class SMACEnv(gym.Env):
    env_name = "SMAC"

    def __init__(self, cfg):
        self.env = StarCraft2Env(cfg)

        policy_obs_dim = self.env.observation_space[0][0]

        policy_obs = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(policy_obs_dim,), dtype=np.float32
        )

        critic_obs_dim = self.env.share_observation_space[0][0]

        critic_obs = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(critic_obs_dim,), dtype=np.float32
        )

        self.agent_num = self.env.n_agents
        self.observation_space = gym.spaces.Dict(
            {
                "policy": policy_obs,
                "critic": critic_obs,
            }
        )

        self.action_space = self.env.action_space[0]

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.env.seed(seed)
        local_obs, global_state, action_masks = self.env.reset()

        return {"policy": local_obs, "critic": global_state}, {
            "action_masks": action_masks
        }

    def step(self, action):
        (
            local_obs,
            global_state,
            rewards,
            dones,
            infos,
            action_mask,
        ) = self.env.step(action)
        infos.update({"action_masks": action_mask})
        return (
            {"policy": local_obs, "critic": global_state},
            rewards,
            dones,
            infos,
        )

    def close(self):
        self.env.close()
