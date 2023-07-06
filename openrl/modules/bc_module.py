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

from openrl.modules.model_config import ModelTrainConfig
from openrl.modules.networks.policy_network import PolicyNetwork
from openrl.modules.ppo_module import PPOModule


class BCModule(PPOModule):
    def get_model_configs(self, cfg) -> Dict[str, Any]:
        model_configs = {
            "policy": ModelTrainConfig(
                lr=cfg.lr,
                model=(
                    self.model_dict["policy"]
                    if self.model_dict and "policy" in self.model_dict
                    else PolicyNetwork
                ),
                input_space=self.policy_input_space,
            )
        }
        return model_configs
