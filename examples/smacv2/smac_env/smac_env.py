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

from .wrapper import StarCraftCapabilityEnvWrapper


class SMACEnv(gym.Env):
    env_name = "SMAC"

    def __init__(self, cfg):
        self.env = StarCraftCapabilityEnvWrapper(
            cfg=cfg, capability_config=self.parse_smacv2_distribution(cfg)
        )

        action_space = []
        observation_space = []
        share_observation_space = []

        for i in range(self.env.n_agents):
            action_space.append(gym.spaces.Discrete(self.env.env.n_actions))
            observation_space.append(self.env.env.get_obs_size())
            share_observation_space.append(self.env.env.get_state_size())

        policy_obs_dim = observation_space[0][0]

        policy_obs = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(policy_obs_dim,), dtype=np.float32
        )

        critic_obs_dim = share_observation_space[0][0]

        critic_obs = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(critic_obs_dim,), dtype=np.float32
        )

        self.agent_num = self.env.env.n_agents
        self.observation_space = gym.spaces.Dict(
            {
                "policy": policy_obs,
                "critic": critic_obs,
            }
        )

        self.action_space = action_space[0]

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

    def parse_smacv2_distribution(self, args):
        units = args.units.split("v")
        distribution_config = {
            "n_units": int(units[0]),
            "n_enemies": int(units[1]),
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": 32,
                "map_y": 32,
            },
        }
        if "protoss" in args.map_name:
            distribution_config["team_gen"] = {
                "dist_type": "weighted_teams",
                "unit_types": ["stalker", "zealot", "colossus"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            }
        elif "zerg" in args.map_name:
            distribution_config["team_gen"] = {
                "dist_type": "weighted_teams",
                "unit_types": ["zergling", "baneling", "hydralisk"],
                "weights": [0.45, 0.1, 0.45],
                "observe": True,
            }
        elif "terran" in args.map_name:
            distribution_config["team_gen"] = {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            }
        return distribution_config
