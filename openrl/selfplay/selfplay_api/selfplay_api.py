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
import time

import trueskill
from ray import serve

from openrl.selfplay.sample_strategy import SampleStrategyFactory
from openrl.selfplay.selfplay_api.base_api import (
    BaseSelfplayAPIServer,
    BattleData,
    OpponentData,
    OpponentModel,
    SampleStrategyData,
    SkillData,
    app,
)
from openrl.selfplay.selfplay_api.opponent_model import BattleResult


@serve.deployment(route_prefix="/selfplay")
@serve.ingress(app)
class SelfplayAPIServer(BaseSelfplayAPIServer):
    @app.post("/set_sample_strategy")
    async def set_sample_strategy(self, sample_strategy_data: SampleStrategyData):
        try:
            self.sample_strategy = SampleStrategyFactory.get_sample_strategy(
                sample_strategy_data.sample_strategy
            )()
        except KeyError:
            return {
                "success": False,
                "error": (
                    f"Sample strategy {sample_strategy_data.sample_strategy} not found."
                ),
            }

        return {"success": True}

    @app.post("/add_opponent")
    async def add_opponent(self, opponent_data: OpponentData):
        opponent_id = opponent_data.opponent_id
        self.opponents.append(
            opponent_id,
            OpponentModel(
                opponent_id,
                opponent_path=opponent_data.opponent_info["opponent_path"],
                opponent_info=opponent_data.opponent_info,
            ),
        )
        return {
            "add_opponent reponse msg": (
                f"Opponent {opponent_id} added with model path:"
                f" {opponent_data.opponent_info['opponent_path']}"
            )
        }

    @app.get("/get_opponent")
    async def get_opponent(self):
        while self.sample_strategy is None:
            time.sleep(1)

        opponent = self.sample_strategy.sample_opponent(self.opponents)
        return {
            "opponent_id": opponent.opponent_id,
            "opponent_path": opponent.opponent_path,
            "opponent_type": opponent.opponent_type,
        }

    @app.post("/add_battle_result")
    async def add_battle_result(self, battle_data: BattleData):
        battle_info = battle_data.battle_info

        assert "winner_ids" in battle_info, "battle_info must contain winners"
        assert "loser_ids" in battle_info, "battle_info must contain losers"
        the_winner = None
        the_loser = None
        drawn = False

        if len(battle_info["loser_ids"]) == 0:
            # draw
            drawn = True
            assert "training_agent" in battle_info["winner_ids"]
            for winner_id in battle_info["winner_ids"]:
                if winner_id == "training_agent":
                    the_winner = self.training_agent
                    the_winner.add_battle_result(BattleResult.DRAW)
                else:
                    the_loser = self.opponents[winner_id]
                    the_loser.add_battle_result(BattleResult.DRAW)

        else:
            # win
            for winner_id in battle_info["winner_ids"]:
                if winner_id == "training_agent":
                    the_winner = self.training_agent
                else:
                    the_winner = self.opponents[winner_id]
                the_winner.add_battle_result(BattleResult.WIN)
            # lose
            for loser_id in battle_info["loser_ids"]:
                if loser_id == "training_agent":
                    the_loser = self.training_agent
                else:
                    the_loser = self.opponents[loser_id]
                the_loser.add_battle_result(BattleResult.LOSE)

        # update trueskill
        the_winner.rating, the_loser.rating = trueskill.rate_1vs1(
            the_winner.rating, the_loser.rating, drawn=drawn
        )

        return {"success": True}
