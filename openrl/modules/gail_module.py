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

from openrl.buffers.utils.util import get_shape_from_act_space
from openrl.modules.model_config import ModelTrainConfig
from openrl.modules.networks.gail_discriminator import Discriminator
from openrl.modules.ppo_module import PPOModule


class GAILModule(PPOModule):
    def get_model_configs(self, cfg) -> Dict[str, Any]:
        model_configs = super().get_model_configs(cfg)

        if cfg.dis_input_len is None:
            assert (
                len(self.critic_input_space.shape) == 1
            ), "critic_input_space must be 1D, but got {}".format(
                self.critic_input_space.shape
            )

            if cfg.gail_use_action:
                gail_input_space = self.critic_input_space.shape[
                    0
                ] + get_shape_from_act_space(self.act_space)
            else:
                gail_input_space = self.critic_input_space.shape[0]
        else:
            gail_input_space = cfg.dis_input_len

        model_configs["gail_discriminator"] = ModelTrainConfig(
            lr=cfg.gail_lr,
            model=(
                self.model_dict["gail_discriminator"]
                if self.model_dict and "gail_discriminator" in self.model_dict
                else Discriminator
            ),
            input_space=gail_input_space,
        )
        return model_configs
