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
        self.player_ids = None

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
            self.opponent = self.get_opponent(self.opponent_players)
            if self.opponent is not None:
                self.opponent.reset()

        if self.opponent is None:
            mask = observation["action_mask"]
            action_space = self.env.action_space(player_name)
            if isinstance(action_space, list):
                action = []
                for space in action_space:
                    action.append(space.sample(mask))
            else:
                action = action_space.sample(mask)
        else:
            action = self.opponent.act(
                player_name, observation, reward, termination, truncation, info
            )
        return action

    def on_episode_end(
        self, player_name, observation, reward, termination, truncation, info
    ):
        assert "winners" in info, "winners must be in info"
        assert "losers" in info, "losers must be in info"
        assert len(info["winners"]) >= 1, "winners must be at least 1"

        winner_ids = []
        loser_ids = []

        for player in info["winners"]:
            if player == self.self_player:
                winner_id = "training_agent"
            else:
                winner_id = self.opponent.opponent_id
            winner_ids.append(winner_id)

        for player in info["losers"]:
            if player == self.self_player:
                loser_id = "training_agent"
            else:
                loser_id = self.opponent.opponent_id
            loser_ids.append(loser_id)
        assert set(winner_ids).isdisjoint(set(loser_ids)), (
            "winners and losers must be disjoint, but get winners: {}, losers: {}"
            .format(winner_ids, loser_ids)
        )
        battle_info = {"winner_ids": winner_ids, "loser_ids": loser_ids}
        self.api_client.add_battle_result(battle_info)
