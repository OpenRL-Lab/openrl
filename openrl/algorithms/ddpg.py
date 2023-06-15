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

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.modules.networks.utils.distributed_utils import reduce_tensor
from openrl.modules.utils.util import get_gard_norm, huber_loss, mse_loss
from openrl.utils.util import check

class DDPGAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            cfg,
            init_module,
            agent_num: int = 1,
            device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(cfg, init_module, agent_num, device)

        self.gamma = cfg.gamma
        self.target_update = cfg.target_update
        self.counter = 0

    def ddpg_update(self, sample, turn_on=True):
        pass