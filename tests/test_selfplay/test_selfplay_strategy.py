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
import os
import sys
import pytest

from openrl.selfplay.strategies import NaiveSelfplayStrategy
from openrl.selfplay.strategies import OnlyLatestSelfplayStrategy
from openrl.selfplay.strategies import WeightSelfplayStrategy
from openrl.selfplay.strategies import WinRateSelfplayStrategy
from openrl.selfplay.strategies import VarExistEnemySelfplayStrategy
from openrl.selfplay.strategies import WeightExistEnemySelfplayStrategy


@pytest.mark.unittest
def test_naive_selfplay():
    strategy = NaiveSelfplayStrategy()
    strategy.get_plist()
    strategy.update_weight()
    strategy.update_win_rate()
    strategy.push_newone()
    

@pytest.mark.unittest
def test_only_latest_selfplay():
    strategy = OnlyLatestSelfplayStrategy()
    strategy.get_plist()
    strategy.update_weight()
    strategy.update_win_rate()
    strategy.push_newone()


@pytest.mark.unittest
def test_weight_selfplay():
    strategy = WeightSelfplayStrategy()
    strategy.get_plist()
    strategy.update_weight()
    strategy.update_win_rate()
    strategy.push_newone()


@pytest.mark.unittest
def test_win_rate_selfplay():
    strategy = WinRateSelfplayStrategy()
    strategy.get_plist()
    strategy.update_weight()
    strategy.update_win_rate()
    strategy.push_newone()


@pytest.mark.unittest
def test_var_exist_enemy_selfplay():
    strategy = VarExistEnemySelfplayStrategy()
    strategy.get_plist()
    strategy.update_weight()
    strategy.update_win_rate()
    strategy.push_newone()


@pytest.mark.unittest
def test_weight_exist_enemy_selfplay():
    strategy = WeightExistEnemySelfplayStrategy()
    strategy.get_plist()
    strategy.update_weight()
    strategy.update_win_rate()
    strategy.push_newone()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))