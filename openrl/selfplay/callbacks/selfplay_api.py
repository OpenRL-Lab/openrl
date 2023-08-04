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
from typing import Optional

from ray import serve

from openrl.selfplay.callbacks.base_callback import BaseSelfplayCallback
from openrl.selfplay.selfplay_api.selfplay_api import SelfplayAPIServer
from openrl.selfplay.selfplay_api.selfplay_client import SelfPlayClient


class SelfplayAPI(BaseSelfplayCallback):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10086,
        sample_strategy: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.host = host
        self.port = port
        self.sample_strategy = sample_strategy
        self.api_client = SelfPlayClient(host=host, port=port)

    def _init_callback(self) -> None:
        serve.start(
            http_options={
                "location": "EveryNode",
                "host": self.host,
                "port": self.port,
            },
            detached=True,
        )

        self.bind = SelfplayAPIServer.bind()
        serve.run(self.bind)
        success = False
        try_time = 10
        while not success:
            success = self.api_client.set_sample_strategy(self.sample_strategy)
            try_time -= 1
            if try_time <= 0:
                raise RuntimeError("Failed to set sample strategy.")

    def _on_step(self) -> bool:
        # print("To send request to API server.")
        # response = requests.get("http://localhost:52365/api/serve/deployments/")
        # status_info = response.json()
        # print(status_info)
        return True

    def _on_training_end(self) -> None:
        application_name = "SelfplayAPIServer"
        if self.verbose >= 2:
            print(f"deleting {application_name}")
        serve.delete(application_name)
        del self.bind
        if self.verbose >= 2:
            print(f"delete {application_name} done!")
