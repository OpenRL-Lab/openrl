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

from openrl.selfplay.strategies import (
    NaiveSelfplayStrategy,
    OnlyLatestSelfplayStrategy,
    VarExistEnemySelfplayStrategy,
    WeightExistEnemySelfplayStrategy,
    WeightSelfplayStrategy,
    WinRateSelfplayStrategy,
)


@pytest.fixture(scope="module", params=[""])
def config(request):
    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_naive_selfplay(config):
    strategy = NaiveSelfplayStrategy(config, 1, 1)
    strategy.get_plist()
    strategy.update_weight(enemy_loses=1)
    strategy.update_win_rate(dones=True, enemy_wins=1)
    strategy.push_newone()


@pytest.mark.unittest
def test_only_latest_selfplay(config):
    strategy = OnlyLatestSelfplayStrategy(config, 1, 1)
    strategy.get_plist()
    strategy.update_weight(enemy_loses=1)
    # strategy.update_win_rate(enemy_wins=0,
    #                          enemy_ties=0,
    #                          enemy_loses=0)
    strategy.push_newone()


@pytest.mark.unittest
def test_weight_selfplay(config):
    strategy = WeightSelfplayStrategy(config, 1, 1)
    strategy.get_plist()
    strategy.update_weight(enemy_loses=1)
    # strategy.update_win_rate(dones=True,
    #                          enemy_wins=1)
    strategy.push_newone()


@pytest.mark.unittest
def test_win_rate_selfplay(config):
    strategy = WinRateSelfplayStrategy(config, 1, 1)
    strategy.get_plist()
    strategy.update_weight(enemy_loses=1)
    # strategy.update_win_rate(enemy_wins=1,
    #                          enemy_ties=1,
    #                          enemy_loses=1)
    # strategy.push_newone()


@pytest.mark.unittest
def test_var_exist_enemy_selfplay(config):
    strategy = VarExistEnemySelfplayStrategy(config, 1, 1)
    strategy.get_plist()
    strategy.update_weight(enemy_loses=1)
    # strategy.update_win_rate(enemy_wins=1,
    #                          enemy_ties=1,
    #                          enemy_loses=1)
    strategy.push_newone()


@pytest.mark.unittest
def test_weight_exist_enemy_selfplay(config):
    strategy = WeightExistEnemySelfplayStrategy(config, 1, 1)
    strategy.get_plist()
    strategy.update_weight(enemy_loses=1)
    # strategy.update_win_rate(enemy_wins=1,
    #                          enemy_ties=1,
    #                          enemy_loses=1)
    strategy.push_newone()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
