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

from typing import Any, Dict, Optional, Union

import torch

from openrl.envs.vec_env.wrappers.reward_wrapper import RewardWrapper
from openrl.modules.base_module import BaseModule
from openrl.modules.common.ppo_net import PPONet
from openrl.modules.gail_module import GAILModule
from openrl.rewards.gail_reward import GAILReward


class GAILNet(PPONet):
    def __init__(
        self,
        env: RewardWrapper,
        cfg=None,
        device: Union[torch.device, str] = "cpu",
        n_rollout_threads: int = 1,
        model_dict: Optional[Dict[str, Any]] = None,
        module_class: type(BaseModule) = GAILModule,
    ) -> None:
        super().__init__(env, cfg, device, n_rollout_threads, model_dict, module_class)
        assert isinstance(
            env.reward_class, GAILReward
        ), "env.reward_class must be GAILReward, but got {}".format(env.reward_class)
        env.reward_class.set_discriminator(
            self.cfg, self.module.models["gail_discriminator"]
        )

        self.env = env
