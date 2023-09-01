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
import copy
from typing import Optional

from openrl.selfplay.wrappers.base_multiplayer_wrapper import BaseMultiPlayerWrapper


class HumanOpponentWrapper(BaseMultiPlayerWrapper):
    def get_opponent_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        action = self.env.get_human_action(
            player_name, observation, termination, truncation, info
        )
        action = [action]
        return action
