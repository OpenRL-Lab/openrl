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
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import gymnasium
import numpy as np

from openrl.selfplay.opponents.base_opponent import BaseOpponent


class JiDiOpponent(BaseOpponent):
    def __init__(
        self,
        opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None,
        jidi_controller: Optional[Callable] = None,
        player_num: int = 1,
    ):
        self.player_num = player_num
        self.jidi_controller = jidi_controller
        super().__init__(opponent_id, opponent_path, opponent_info)

    def act(self, player_name, observation, reward, termination, truncation, info):
        # if self.player_num == 1:
        #     observation = [observation]
        # else:
        assert len(observation) == self.player_num

        joint_action = []
        for i in range(self.player_num):
            action = self.jidi_controller(
                observation[i], self.action_space_list[i], self.is_act_continuous
            )
            if isinstance(self.action_space_list[i][0], gymnasium.spaces.Discrete):
                action = np.argmax(action[0])
            else:
                action = action[0]

            joint_action.append(action)

        return joint_action

    def _load(self, opponent_path: Union[str, Path]):
        pass

    def _set_env(self, env, opponent_player: str):
        self.action_space_list = env.action_space(opponent_player)

        assert len(self.action_space_list) == self.player_num

        self.is_act_continuous = self.action_space_list[0].__class__.__name__ == "Box"

        for i in range(self.player_num):
            self.action_space_list[i] = [self.action_space_list[i]]
