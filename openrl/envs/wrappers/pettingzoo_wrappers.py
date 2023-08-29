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
from collections import defaultdict
from typing import Dict, Optional

from pettingzoo.utils.env import ActionType, AECEnv
from pettingzoo.utils.wrappers import BaseWrapper


class SeedEnv(BaseWrapper):
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            for i, space in enumerate(
                list(self.action_spaces.values())
                + list(self.observation_spaces.values())
            ):
                if isinstance(space, list):
                    for j in range(len(space)):
                        space[j].seed(seed + i * 7891 + j)
                else:
                    space.seed(seed + i * 7891)


class RecordWinner(BaseWrapper):
    def __init__(self, env: AECEnv):
        super().__init__(env)
        self.total_rewards = defaultdict(float)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.total_rewards = defaultdict(float)
        return super().reset(seed, options)

    def step(self, action: ActionType) -> None:
        super().step(action)
        winners = None
        losers = None
        for agent in self.terminations:
            if self.terminations[agent]:
                if winners is None:
                    winners = self.get_winners()
                    losers = [player for player in self.agents if player not in winners]
                self.infos[agent]["winners"] = winners
                self.infos[agent]["losers"] = losers

    def get_winners(self):
        max_reward = max(self.total_rewards.values())

        winners = [
            agent
            for agent, reward in self.total_rewards.items()
            if reward == max_reward
        ]
        return winners

    def last(self, observe: bool = True):
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection
        # if self._cumulative_rewards[agent]!=0:
        #     print("agent:",agent,self._cumulative_rewards[agent])
        self.total_rewards[agent] += self._cumulative_rewards[agent]

        return super().last(observe)
