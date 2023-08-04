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

from ray import serve

from openrl.selfplay.sample_strategy import SampleStrategyFactory
from openrl.selfplay.selfplay_api.base_api import (
    BaseSelfplayAPIServer,
    OpponentData,
    OpponentModel,
    SampleStrategyData,
    SkillData,
    app,
)


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
            OpponentModel(
                opponent_id,
                opponent_path=opponent_data.opponent_info["opponent_path"],
                opponent_info=opponent_data.opponent_info,
            )
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

        opponent_index = self.sample_strategy.sample_opponent(self.opponents)
        return {
            "opponent_id": self.opponents[opponent_index].opponent_id,
            "opponent_path": self.opponents[opponent_index].opponent_path,
            "opponent_type": self.opponents[opponent_index].opponent_type,
        }

    @app.post("/update_skill")
    async def update_skill(self, data: SkillData):
        self.opponents[data.opponent_id].update_skill(
            self.opponents[data.other_id], data.result
        )
        return {"msg": "Skill updated."}
