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
from collections import deque
from typing import Any, Dict

import numpy as np

from openrl.envs.vec_env.vec_info.simple_vec_info import SimpleVecInfo


class EPS_RewardInfo(SimpleVecInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_rewards = deque(maxlen=100)

    def statistics(self, buffer: Any) -> Dict[str, Any]:
        info_dict = super().statistics(buffer)
        for step_info in self.infos:
            for singe_env_info in step_info:
                assert isinstance(singe_env_info, dict), "singe_env_info must be dict"

                if (
                    "final_info" in singe_env_info.keys()
                    and "episode" in singe_env_info["final_info"].keys()
                ):
                    self.episode_rewards.append(
                        singe_env_info["final_info"]["episode"]["r"]
                    )

        if len(self.episode_rewards) > 0:
            info_dict["episode_rewards_mean"] = np.mean(self.episode_rewards)
            info_dict["episode_rewards_median"] = np.median(self.episode_rewards)
            info_dict["episode_rewards_min"] = np.min(self.episode_rewards)
            info_dict["episode_rewards_max"] = np.max(self.episode_rewards)

        return info_dict
