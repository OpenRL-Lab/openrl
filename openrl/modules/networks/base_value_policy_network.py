#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 The OpenRL Authors.
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
from abc import ABC, abstractmethod

import torch.nn as nn

from openrl.modules.utils.valuenorm import ValueNorm


class BaseValuePolicyNetwork(ABC, nn.Module):
    def __init__(self, cfg, device):
        super(BaseValuePolicyNetwork, self).__init__()
        self.device = device
        self._use_valuenorm = cfg.use_valuenorm

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.get_actions(*args, **kwargs)
        elif forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        elif forward_type == "get_values":
            return self.get_values(*args, **kwargs)
        else:
            raise NotImplementedError

    @abstractmethod
    def get_actions(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def eval_actions(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_values(self, *args, **kwargs):
        raise NotImplementedError

    def get_actor_para(self):
        return self.parameters()

    def get_critic_para(self):
        return self.parameters()
