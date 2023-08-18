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
from typing import Optional

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
                space.seed(seed + i * 7891)


class RecordWinner(BaseWrapper):
    def __init__(self, env: AECEnv):
        super().__init__(env)
        self.cumulative_rewards = {}

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
        max_reward = max(self._cumulative_rewards.values())

        winners = [
            agent
            for agent, reward in self._cumulative_rewards.items()
            if reward == max_reward
        ]
        return winners
