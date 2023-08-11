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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union


class BaseOpponent(ABC):
    def __init__(
        self,
        opponent_id: Optional[str] = None,
        opponent_path: Optional[Union[str, Path]] = None,
        opponent_info: Optional[Dict[str, str]] = None,
    ):
        self.opponent_id = opponent_id
        self.opponent_path = opponent_path
        self.opponent_info = opponent_info

        if opponent_path is not None:
            self.load(opponent_path)

    @property
    def opponent_type(self):
        return self.opponent_info["opponent_type"]

    def reset(self, env=None, opponent_player: Optional[str] = None):
        if env is not None:
            self.set_env(env, opponent_player)

    def set_env(self, env, opponent_player: Optional[str] = None):
        self.env = env
        self._set_env(env, opponent_player)

    def _set_env(self, env, opponent_player: Optional[str] = None):
        pass

    def load(
        self, opponent_path: Union[str, Path], opponent_id: Optional[str] = None
    ) -> "BaseOpponent":
        self.opponent_path = opponent_path
        if opponent_id is not None:
            self.opponent_id = opponent_id
        self._load(opponent_path)
        return self

    @abstractmethod
    def _load(self, opponent_path: Union[str, Path]):
        raise NotImplementedError()

    @abstractmethod
    def act(self, player_name, observation, reward, termination, truncation, info):
        pass

    def log(self):
        print(
            f"Opponent info: opponent_id: {self.opponent_id}, opponent_path:"
            f" {self.opponent_path}, opponent_type: {self.opponent_type}"
        )
