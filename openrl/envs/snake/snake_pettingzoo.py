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
import functools
from copy import deepcopy
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from openrl.envs.snake.snake import SnakeEatBeans

NONE = 4


class SnakeEatBeansAECEnv(AECEnv):
    metadata = {"render.modes": ["human"], "name": "SnakeEatBeans"}

    @property
    def agent_num(self):
        return self.player_each_side

    def __init__(self, render_mode: Optional[str] = None, id: str = None):
        self.env = SnakeEatBeans(render_mode, id=id)

        agent_num = len(self.possible_agents)
        player_each_side = self.env.num_agents
        self.player_each_side = player_each_side
        self.agent_name_to_slice = dict(
            zip(
                self.possible_agents,
                [
                    slice(i * player_each_side, (i + 1) * player_each_side)
                    for i in range(agent_num)
                ],
            )
        )
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {
            agent: [spaces.Discrete(4) for _ in range(self.env.num_agents)]
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(288,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.agents = self.possible_agents[:]

        self.observations = {agent: NONE for agent in self.agents}
        self.raw_obs, self.raw_reward, self.raw_done, self.raw_info = (
            None,
            None,
            None,
            None,
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return deepcopy(self._observation_spaces[agent])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return deepcopy(self._action_spaces[agent])

    def observe(self, agent):
        obs = self.raw_obs[self.agent_name_to_slice[agent]]
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self.env.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}

        self.raw_obs, self.raw_info = self.env.reset()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            joint_action = []
            for agent in self.agents:
                joint_action.append(self.state[agent])

            joint_action = np.concatenate(joint_action)

            self.raw_obs, self.raw_reward, self.raw_done, self.raw_info = self.env.step(
                joint_action
            )

            self.rewards = {
                agent: np.sum(self.raw_reward[self.agent_name_to_slice[agent]])
                for agent in self.agents
            }

            if np.any(self.raw_done):
                for key in self.terminations:
                    self.terminations[key] = True
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            self._clear_rewards()

            # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        img = self.env.render()
        return img

    def close(self):
        self.env.close()

    @property
    def possible_agents(self):
        return ["player_" + str(i) for i in range(2)]

    @property
    def num_agents(self):
        return len(self.possible_agents)
