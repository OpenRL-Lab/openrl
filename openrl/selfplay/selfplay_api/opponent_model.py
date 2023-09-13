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
from enum import Enum
from typing import Any, Dict, List, Optional

import trueskill


class BattleResult(Enum):
    WIN = 0
    LOSE = 1
    DRAW = 2


class BattleHistory:
    def __init__(self):
        self.num_games = 0
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0
        self.battle_results: List[BattleResult] = []

    def update(self, result: BattleResult):
        self.num_games += 1
        if result == BattleResult.WIN:
            self.num_wins += 1
        elif result == BattleResult.LOSE:
            self.num_losses += 1
        else:
            self.num_draws += 1
        self.battle_results.append(result)

    def get_battle_info(self) -> Dict[str, Any]:
        result = {}
        result["win_rate"] = float(self.num_wins) / max(self.num_games, 1)
        result["draw_rate"] = float(self.num_draws) / max(self.num_games, 1)
        result["loss_rate"] = float(self.num_losses) / max(self.num_games, 1)
        result["total_games"] = self.num_games
        return result


class OpponentModel:
    def __init__(
        self,
        opponent_id: str,
        opponent_path: Optional[str] = None,
        opponent_info: Optional[Dict[str, str]] = None,
    ):
        self.opponent_id = opponent_id
        self.opponent_path = opponent_path
        self.opponent_info = opponent_info
        self.batch_history = BattleHistory()
        self.rating = trueskill.Rating()

    @property
    def opponent_type(self):
        return self.opponent_info["opponent_type"]

    def add_battle_result(self, result: BattleResult):
        self.batch_history.update(result)
