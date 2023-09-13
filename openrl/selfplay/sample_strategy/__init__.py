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
from openrl.selfplay.sample_strategy.last_opponent import LastOpponent
from openrl.selfplay.sample_strategy.random_opponent import RandomOpponent

sample_strategy_dict = {
    "LastOpponent": LastOpponent,
    "RandomOpponent": RandomOpponent,
}


class SampleStrategyFactory:
    def __init__(self):
        pass

    @staticmethod
    def register_sample_strategy(name, sample_strategy):
        sample_strategy_dict[name] = sample_strategy

    @staticmethod
    def get_sample_strategy(name):
        return sample_strategy_dict[name]
