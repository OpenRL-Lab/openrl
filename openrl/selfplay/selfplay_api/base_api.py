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
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from openrl.selfplay.selfplay_api.opponent_model import OpponentModel
from openrl.utils.custom_data_structure import ListDict


class OpponentData(BaseModel):
    opponent_id: str
    opponent_info: Dict[str, str]


class SkillData(BaseModel):
    opponent_id: str
    other_id: str
    result: int


class SampleStrategyData(BaseModel):
    sample_strategy: str


class BattleData(BaseModel):
    battle_info: Dict[str, Any]


app = FastAPI()


class BaseSelfplayAPIServer(ABC):
    def __init__(self):
        logger = logging.getLogger("ray.serve")
        logger.setLevel(logging.ERROR)
        self.opponents = ListDict()
        self.training_agent = OpponentModel("training_agent")
        self.sample_strategy = None
