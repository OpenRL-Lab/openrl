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
import requests


class SelfPlayClient:
    def __init__(self, address: str):
        self.address = address

    @staticmethod
    def add_agent(address: str, agent_id: str, agent_info: dict):
        response = requests.post(
            f"{address}add", json={"agent_id": agent_id, "agent_info": agent_info}
        )
        print(response.json())
