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
from typing import Any, Callable, Dict, Optional

from openrl.arena.base_arena import BaseArena
from openrl.arena.games.two_player_game import TwoPlayerGame
from openrl.selfplay.selfplay_api.opponent_model import BattleResult


class TwoPlayerArena(BaseArena):
    def __init__(
        self,
        env_fn: Callable,
        dispatch_func: Optional[Callable] = None,
        use_tqdm: bool = True,
    ):
        super().__init__(env_fn, dispatch_func, use_tqdm=use_tqdm)
        self.game = TwoPlayerGame()

    def _deal_result(self, result: Any):
        if len(result["winners"]) == 2:
            # drawn
            for agent_name in result["winners"]:
                self.agents[agent_name].add_battle_result(BattleResult.DRAW)
        else:
            for agent_name in result["winners"]:
                self.agents[agent_name].add_battle_result(BattleResult.WIN)
            for agent_name in result["losers"]:
                self.agents[agent_name].add_battle_result(BattleResult.LOSE)

    def _get_final_result(self) -> Dict[str, Any]:
        result = {}
        for agent_name in self.agents:
            result[agent_name] = self.agents[agent_name].get_battle_info()

        return result
