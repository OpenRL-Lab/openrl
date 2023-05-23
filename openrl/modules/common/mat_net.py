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

from typing import Any, Dict, Union

import gymnasium as gym
import torch

from openrl.modules.common.ppo_net import PPONet
from openrl.modules.networks.MAT_network import MultiAgentTransformer


class MATNet(PPONet):
    def __init__(
        self,
        env: Union[gym.Env, str],
        cfg=None,
        device: Union[torch.device, str] = "cpu",
        n_rollout_threads: int = 1,
        model_dict: Dict[str, Any] = {"model": MultiAgentTransformer},
    ) -> None:
        cfg.use_share_model = True
        super().__init__(
            env=env,
            cfg=cfg,
            device=device,
            n_rollout_threads=n_rollout_threads,
            model_dict=model_dict,
        )
