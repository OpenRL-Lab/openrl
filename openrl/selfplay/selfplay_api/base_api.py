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

from abc import ABC
from typing import Dict

from fastapi import FastAPI
import trueskill
from pydantic import BaseModel


class AgentData(BaseModel):
    agent_id: str
    agent_info: Dict[str, str]


class SkillData(BaseModel):
    agent_id: str
    other_id: str
    result: int


class Agent:
    def __init__(self, id, model_path=None):
        self.id = id
        self.model_path = model_path
        self.skill = trueskill.Rating()
        self.num_games = 0
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0

    def update_skill(self, other_agent, result):
        new_rating1, new_rating2 = trueskill.rate_1vs1(
            self.skill, other_agent.skill, result
        )
        self.skill = new_rating1
        other_agent.skill = new_rating2


app = FastAPI()


class BaseSelfplayAPIServer(ABC):
    def __init__(self):
        self.agents = {}
