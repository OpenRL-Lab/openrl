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


from ray import serve
from openrl.selfplay.selfplay_api.base_api import (
    BaseSelfplayAPIServer,
    app,
    AgentData,
    SkillData,
    Agent,
)


@serve.deployment(route_prefix="/selfplay")
@serve.ingress(app)
class SelfplayAPIServer(BaseSelfplayAPIServer):
    @app.post("/add")
    async def add_agent(self, agent_data: AgentData):
        agent_id = agent_data.agent_id
        self.agents[agent_id] = Agent(
            agent_id, model_path=agent_data.agent_info["model_path"]
        )
        return {
            "msg": f"Agent {agent_id} added with model path: {agent_data.agent_info['model_path']}"
        }

    @app.post("/update_skill")
    async def update_skill(self, data: SkillData):
        self.agents[data.agent_id].update_skill(self.agents[data.other_id], data.result)
        return {"msg": "Skill updated."}
