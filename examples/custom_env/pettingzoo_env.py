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


from rock_paper_scissors import RockPaperScissors
from train_and_test import train_and_test

from openrl.envs.common import make
from openrl.envs.PettingZoo.registration import register
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper

register("RockPaperScissors", RockPaperScissors)

env = make(
    "RockPaperScissors",
    env_num=10,
    opponent_wrappers=[RandomOpponentWrapper],
)

train_and_test(env)
