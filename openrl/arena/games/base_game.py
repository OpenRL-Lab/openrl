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
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from openrl.arena.agents.base_agent import BaseAgent
from openrl.selfplay.opponents.base_opponent import BaseOpponent


class BaseGame(ABC):
    def __init__(self):
        self.dispatch_func = None

    def reset(self, dispatch_func: Optional[Callable] = None):
        if dispatch_func is not None:
            self.dispatch_func = dispatch_func
        else:
            self.dispatch_func = self.default_dispatch_func

    def dispatch_agent_to_player(
        self, players: List[str], agents: Dict[str, BaseAgent]
    ) -> Tuple[Dict[str, BaseOpponent], Dict[str, str]]:
        player2agent = {}
        player2agent_name = self.dispatch_func(players, list(agents.keys()))
        for player in players:
            player2agent[player] = agents[player2agent_name[player]].new_agent()
        return player2agent, player2agent_name

    @staticmethod
    def default_dispatch_func(
        players: List[str], agent_names: List[str]
    ) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def run(self, env_fn, agents):
        raise NotImplementedError
