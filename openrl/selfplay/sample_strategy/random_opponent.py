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

import random

from openrl.selfplay.opponents.base_opponent import BaseOpponent
from openrl.selfplay.sample_strategy.base_sample_strategy import BaseSampleStrategy


class RandomOpponent(BaseSampleStrategy):
    def sample_opponent(self, opponents) -> BaseOpponent:
        opponent_index = random.randint(0, len(opponents) - 1)
        return opponents.get_by_index(opponent_index)
