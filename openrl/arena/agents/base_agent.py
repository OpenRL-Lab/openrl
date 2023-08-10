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
from typing import Any, Dict

from openrl.selfplay.opponents.base_opponent import BaseOpponent
from openrl.selfplay.selfplay_api.opponent_model import BattleHistory, BattleResult


class BaseAgent(ABC):
    def __init__(self):
        self.batch_history = BattleHistory()

    def new_agent(self) -> BaseOpponent:
        agent = self._new_agent()
        return agent

    @abstractmethod
    def _new_agent(self) -> BaseOpponent:
        raise NotImplementedError

    def add_battle_result(self, result: BattleResult):
        self.batch_history.update(result)

    def get_battle_info(self) -> Dict[str, Any]:
        return self.batch_history.get_battle_info()
