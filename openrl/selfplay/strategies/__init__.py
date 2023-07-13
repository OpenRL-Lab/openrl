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
from openrl.selfplay.strategies.strategies import (
    NaiveSelfplayStrategy,
    OnlyLatestSelfplayStrategy,
    VarExistEnemySelfplayStrategy,
    WeightExistEnemySelfplayStrategy,
    WeightSelfplayStrategy,
    WinRateSelfplayStrategy,
)


def make_strategy(strategy_name):
    if strategy_name == "Naive":
        selfplay_strategy = NaiveSelfplayStrategy
    elif strategy_name == "OnlyLatest":
        selfplay_strategy = OnlyLatestSelfplayStrategy
    elif strategy_name == "Weight":
        selfplay_strategy = WeightSelfplayStrategy
    elif strategy_name == "WinRate":
        selfplay_strategy = WinRateSelfplayStrategy
    elif strategy_name == "VarExistEnemy":
        selfplay_strategy = VarExistEnemySelfplayStrategy
    elif strategy_name == "WeightExistEnemy":
        selfplay_strategy = WeightExistEnemySelfplayStrategy
    return selfplay_strategy
