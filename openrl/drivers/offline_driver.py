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

import numpy as np

from openrl.drivers.onpolicy_driver import OnPolicyDriver


class OfflineDriver(OnPolicyDriver):
    def add2buffer(self, data):
        infos = data["infos"]
        offline_actions = []
        for i, info in enumerate(infos):
            if "data_action" not in info:
                assert np.all(data["dones"][i])
                data_action = info["final_info"]["data_action"]
            else:
                data_action = info["data_action"]
            offline_actions.append(data_action)
        offline_actions = np.stack(offline_actions, axis=0)

        data["actions"] = offline_actions
        super().add2buffer(data)
