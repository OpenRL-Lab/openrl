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
from typing import Union

from openrl.selfplay.opponents.base_opponent import BaseOpponent


class RandomOpponent(BaseOpponent):
    def act(self, player_name, observation, reward, termination, truncation, info):
        action = self.sample_random_action(
            player_name, observation, reward, termination, truncation, info
        )
        return action

    def sample_random_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        return self._sample_random_action(
            player_name, observation, reward, termination, truncation, info
        )

    def _sample_random_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        action_space = self.env.action_space(player_name)
        if isinstance(action_space, list):
            if not isinstance(observation, list):
                observation = [observation]

            action = []

            for obs, space in zip(observation, action_space):
                mask = obs.get("action_mask", None)
                action.append(space.sample(mask))
        else:
            mask = observation.get("action_mask", None)
            action = action_space.sample(mask)
        return action

    def _load(self, opponent_path: Union[str, Path]):
        pass

    def _set_env(self, env, opponent_player: str):
        pass
