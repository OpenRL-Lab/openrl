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

"""
Play with self.
"""
import copy
import time
from typing import List, Optional

from openrl.selfplay.opponents.utils import get_opponent_from_info
from openrl.selfplay.selfplay_api.selfplay_client import SelfPlayClient
from openrl.selfplay.wrappers.base_multiplayer_wrapper import BaseMultiPlayerWrapper


class OpponentPoolWrapper(BaseMultiPlayerWrapper):
    def __init__(self, env, cfg, reward_class=None) -> None:
        super().__init__(env, cfg, reward_class)

        host = cfg.selfplay_api.host
        port = cfg.selfplay_api.port
        self.api_client = SelfPlayClient(f"http://{host}:{port}/selfplay/")
        self.opponent = None
        self.opponent_player = None
        self.lazy_load_opponent = cfg.lazy_load_opponent

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        results = super().reset(seed=seed, **kwargs)
        self.opponent = self.get_opponent(self.opponent_players)
        if self.opponent is not None:
            self.opponent.reset()
        return results

    def get_opponent(self, opponent_players: List[str]):
        opponent_info = self.api_client.get_opponent(opponent_players)

        if opponent_info is not None:
            # currentkly, we only support 1 opponent, that means we only support games with two players
            opponent_info = opponent_info[0]
            opponent_player = opponent_players[0]
            opponent, is_new_opponent = get_opponent_from_info(
                opponent_info,
                current_opponent=self.opponent,
                lazy_load_opponent=self.lazy_load_opponent,
            )
            if opponent is None:
                return self.opponent
            if is_new_opponent or (self.opponent_player != opponent_player):
                opponent.set_env(self.env, opponent_player)
                self.opponent_player = opponent_player

            return opponent
        else:
            return self.opponent

    def get_opponent_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        if self.opponent is None:
            mask = observation["action_mask"]
            action = self.env.action_space(player_name).sample(mask)
        else:
            action = self.opponent.act(
                player_name, observation, reward, termination, truncation, info
            )
        return action

    def on_episode_end(
        self, player_name, observation, reward, termination, truncation, info
    ):
        pass
