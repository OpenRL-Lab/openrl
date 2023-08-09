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
from typing import Any, Dict, List, Optional

import requests


class SelfPlayClient:
    def __init__(
        self,
        address: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        if address is not None:
            self.address = address
        else:
            assert host is not None and port is not None
            self.address = f"http://{host}:{port}/selfplay/"

    def add_opponent(self, opponent_id: str, opponent_info: dict):
        response = requests.post(
            f"{self.address}add_opponent",
            json={"opponent_id": opponent_id, "opponent_info": opponent_info},
        )
        return response.json()

    def get_opponent(self, opponent_players: List[str]):
        try:
            request_url = f"{self.address}get_opponent"
            results = []
            for opponent_player in opponent_players:
                # if you want to get different types of opponents for different players, you can pass the player name to the request (TODO)
                response = requests.get(request_url)

                if response.status_code == 404:
                    return None
                else:
                    results.append(response.json())
            return results
        except:
            return None

    def set_sample_strategy(self, sample_strategy: str) -> bool:
        try:
            response = requests.post(
                f"{self.address}set_sample_strategy",
                json={"sample_strategy": sample_strategy},
            )
            if response.status_code == 404:
                return False
            else:
                r = response.json()
                if r["success"]:
                    return True
                else:
                    print(r["error"])
                    return False
        except:
            return False

    def add_battle_result(self, battle_info: Dict[str, Any]) -> bool:
        try:
            response = requests.post(
                f"{self.address}add_battle_result",
                json={"battle_info": battle_info},
            )
            if response.status_code == 404:
                return False
            else:
                r = response.json()
                if r["success"]:
                    return True
                else:
                    print(r["error"])
                    return False
        except:
            return False
