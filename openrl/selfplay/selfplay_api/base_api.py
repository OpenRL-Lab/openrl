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

import logging
from abc import ABC
from typing import Dict, Optional

import trueskill
from fastapi import FastAPI
from pydantic import BaseModel


class OpponentData(BaseModel):
    opponent_id: str
    opponent_info: Dict[str, str]


class SkillData(BaseModel):
    opponent_id: str
    other_id: str
    result: int


class SampleStrategyData(BaseModel):
    sample_strategy: str


class OpponentModel:
    def __init__(
        self,
        opponent_id: str,
        opponent_path: Optional[str] = None,
        opponent_info: Optional[Dict[str, str]] = None,
    ):
        self.opponent_id = opponent_id
        self.opponent_path = opponent_path
        self.opponent_info = opponent_info
        self.skill = trueskill.Rating()
        self.num_games = 0
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0

    @property
    def opponent_type(self):
        return self.opponent_info["opponent_type"]

    def update_skill(self, other_opponent, result):
        new_rating1, new_rating2 = trueskill.rate_1vs1(
            self.skill, other_opponent.skill, result
        )
        self.skill = new_rating1
        other_opponent.skill = new_rating2


app = FastAPI()


class BaseSelfplayAPIServer(ABC):
    def __init__(self):
        logger = logging.getLogger("ray.serve")
        logger.setLevel(logging.ERROR)
        self.opponents = []
        self.sample_strategy = None
