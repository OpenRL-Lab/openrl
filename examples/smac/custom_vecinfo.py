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

from typing import Any, Dict
from functools import reduce
from collections import deque

import numpy as np

from openrl.envs.vec_env.vec_info.episode_rewards_info import EPS_RewardInfo


class SMACInfo(EPS_RewardInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win_history = deque(maxlen=100)

    def statistics(self, buffer: Any) -> Dict[str, Any]:
        info_dict = super().statistics(buffer)

        for step_info in self.infos:
            for singe_env_info in step_info:
                assert isinstance(singe_env_info, dict), "singe_env_info must be dict"

                if "final_info" in singe_env_info.keys():
                    assert (
                        "game_state" in singe_env_info["final_info"].keys()
                    ), "game_state must be in info"
                    assert singe_env_info["final_info"]["game_state"] in [
                        "win",
                        "lose",
                    ], "game_state in the final_info must be win or lose"
                    self.win_history.append(
                        singe_env_info["final_info"]["game_state"] == "win"
                    )

        if len(self.win_history) > 0:
            info_dict["win_rate"] = np.mean(self.win_history)

        dead_ratio = 1 - buffer.data.active_masks.sum() / reduce(
            lambda x, y: x * y, list(buffer.data.active_masks.shape)
        )
        info_dict["dead_ratio"] = dead_ratio

        return info_dict
