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
from typing import (
    Any,
    Dict,
    Optional,
    Union,
)

import torch
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from openrl.envs.vec_env import BaseVecEnv


class Isaac2OpenRLWrapper:
    def __init__(self, env: VecEnvRLGames) -> BaseVecEnv:
        self.env = env

    @property
    def parallel_env_num(self) -> int:
        return self.env.num_envs

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        return self.env.action_space

    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        return self.env.observation_space

    def reset(self, **kwargs):
        """Reset all environments."""
        obs_dict = self.env.reset()
        return obs_dict["obs"].unsqueeze(1).cpu().numpy()

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""

        actions = torch.from_numpy(actions).squeeze(-1)

        obs_dict, self._rew, self._resets, self._extras = self.env.step(actions)

        obs = obs_dict["obs"].unsqueeze(1).cpu().numpy()
        rewards = self._rew.unsqueeze(-1).unsqueeze(-1).cpu().numpy()
        dones = self._resets.unsqueeze(-1).cpu().numpy().astype(bool)

        infos = []
        for i in range(dones.shape[0]):
            infos.append({})

        return obs, rewards, dones, infos

    def close(self, **kwargs):
        return self.env.close()

    @property
    def agent_num(self):
        return 1

    @property
    def use_monitor(self):
        return False

    @property
    def env_name(self):
        return "Isaac-" + self.env._task.name

    def batch_rewards(self, buffer):
        return {}
