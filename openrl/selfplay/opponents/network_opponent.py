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
from typing import Dict, Optional, Union

from openrl.selfplay.opponents.base_opponent import BaseOpponent


class NetworkOpponent(BaseOpponent):
    def __init__(
        self,
        opponent_id: str,
        opponent_path: Union[str, Path],
        opponent_info: Dict[str, str],
    ):
        self.agent = None
        self.opponent_env = None
        self.deterministic_action = False
        super().__init__(opponent_id, opponent_path, opponent_info)

    def reset(self, env=None, opponent_player: Optional[str] = None):
        super().reset(env, opponent_player)
        if self.opponent_env is not None:
            self.opponent_env.reset()
        if self.agent is not None:
            self.agent.reset()

    def _load(self, opponent_path: Union[str, Path]):
        model_path = Path(opponent_path) / "module.pt"
        if self.agent is not None:
            self.agent.load(model_path)

    def _set_env(self, env, opponent_player: Optional[str] = None):
        pass

    def act(self, player_name, observation, reward, termination, truncation, info):
        observation, termination, truncation, info = self.opponent_env.process_obs(
            observation, termination, truncation, info
        )
        action, _ = self.agent.act(
            observation, info, deterministic=self.deterministic_action
        )
        action = self.opponent_env.process_action(action)
        return action
