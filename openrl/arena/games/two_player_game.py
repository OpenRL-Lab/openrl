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
import random
from typing import Callable, Dict, List

import numpy as np

from openrl.arena.agents.base_agent import BaseAgent
from openrl.arena.games.base_game import BaseGame


class TwoPlayerGame(BaseGame):
    @staticmethod
    def default_dispatch_func(
        np_random: np.random.Generator,
        players: List[str],
        agent_names: List[str],
    ) -> Dict[str, str]:
        assert len(players) == len(
            agent_names
        ), "The number of players must be equal to the number of agents."
        assert len(players) == 2, "The number of players must be equal to 2."
        np_random.shuffle(agent_names)
        return dict(zip(players, agent_names))

    def _run(self, env_fn: Callable, agents: List[BaseAgent]):
        env = env_fn()
        env.reset(seed=self.seed)

        player2agent, player2agent_name = self.dispatch_agent_to_player(
            env.agents, agents
        )

        for player, agent in player2agent.items():
            agent.reset(env, player)
        result = {}
        while True:
            termination = False
            info = {}
            for player_name in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()

                if termination:
                    break
                action = player2agent[player_name].act(
                    player_name, observation, reward, termination, truncation, info
                )
                env.step(action)

            if termination:
                assert "winners" in info, "The game is terminated but no winners."
                assert "losers" in info, "The game is terminated but no losers."

                result["winners"] = [
                    player2agent_name[player] for player in info["winners"]
                ]
                result["losers"] = [
                    player2agent_name[player] for player in info["losers"]
                ]
                break
        env.close()
        return result
