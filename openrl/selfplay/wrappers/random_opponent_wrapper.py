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
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from openrl.selfplay.wrappers.base_multiplayer_wrapper import BaseMultiPlayerWrapper


class RandomOpponentWrapper(BaseMultiPlayerWrapper):
    def get_opponent_action(self, agent, observation, termination, truncation, info):
        mask = observation["action_mask"]
        action = self.env.action_space(agent).sample(mask)
        return action
