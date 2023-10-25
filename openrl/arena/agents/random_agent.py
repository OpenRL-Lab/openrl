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
from openrl.arena.agents.base_agent import BaseAgent
from openrl.selfplay.opponents.base_opponent import BaseOpponent
from openrl.selfplay.opponents.random_opponent import RandomOpponent
from openrl.selfplay.opponents.utils import load_opponent_from_path


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def _new_agent(self) -> BaseOpponent:
        return RandomOpponent()
