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
from typing import Dict, List, Optional

from gymnasium.spaces import Discrete
from pettingzoo.utils.env import ActionType, AECEnv
from pettingzoo.utils.wrappers import (
    AssertOutOfBoundsWrapper,
    BaseWrapper,
    OrderEnforcingWrapper,
)

from openrl.envs.wrappers.base_wrapper import BaseWrapper as OpenRLBaseWrapper


class CheckAgentNumber(BaseWrapper, OpenRLBaseWrapper):
    # make the original petting zoo env compatible with openrl
    @property
    def agent_num(self):
        if self.is_original_pettingzoo_env():
            return 1
        else:
            return self.env.agent_num

    def step(self, action: ActionType) -> None:
        if self.is_original_pettingzoo_env():
            action = action[0]
        super().step(action)

    def is_original_pettingzoo_env(self):
        return not hasattr(self.env, "agent_num")

    def action_space(self, agent: str):
        space = self.env.action_space(agent)
        if self.is_original_pettingzoo_env():
            space = [space]
        return space


class SeedEnv(BaseWrapper, OpenRLBaseWrapper):
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


class RecordWinner(BaseWrapper, OpenRLBaseWrapper):
    def __init__(self, env: AECEnv):
        super().__init__(env)
        self.total_rewards = defaultdict(float)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.total_rewards = defaultdict(float)
        return super().reset(seed, options)

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
        # this may be miss the last reward for another agent
        self.total_rewards[agent] += self._cumulative_rewards[agent]

        winners = None
        losers = None
        for agent in self.terminations:
            if self.terminations[agent]:
                if winners is None:
                    winners = self.get_winners()
                    losers = [player for player in self.agents if player not in winners]
                self.infos[agent]["winners"] = winners
                self.infos[agent]["losers"] = losers

        return super().last(observe)


class OpenRLAssertOutOfBoundsWrapper(AssertOutOfBoundsWrapper, OpenRLBaseWrapper):
    """Asserts if the action given to step is outside of the action space."""

    def __init__(self, env: AECEnv):
        BaseWrapper.__init__(self, env)
        assert all(
            isinstance(self.env.action_space(agent), (Discrete, List))
            for agent in getattr(self, "possible_agents", [])
        ), (
            "should only use OpenRLAssertOutOfBoundsWrapper for Discrete spaces or list"
            " of Discrete spaces"
        )

    def step(self, action: ActionType) -> None:
        action_space = self.env.action_space(self.agent_selection)

        if isinstance(action_space, list):
            finished = action is None and (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
            )
            if not finished:
                for space, a in zip(action_space, action):
                    right_action = space.contains(a)

                    assert (
                        right_action
                    ), f"action: {a} is wrong, it should be type of {space}"

            BaseWrapper.step(self, action)
        else:
            super().step(action)


class OpenRLOrderEnforcingWrapper(OrderEnforcingWrapper, OpenRLBaseWrapper):
    pass
