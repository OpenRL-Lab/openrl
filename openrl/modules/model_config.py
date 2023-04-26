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

import gym
import torch


class ModelConfig(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(ModelConfig, self).__init__(*args, **kwargs)


class ModelTrainConfig(ModelConfig):
    def __init__(
        self,
        model: torch.nn.Module,
        input_space: gym.spaces.Box,
        lr: Optional[float] = None,
        *args,
        **kwargs
    ) -> None:
        super(ModelTrainConfig, self).__init__(*args, **kwargs)
        self["model"] = model
        self["input_space"] = input_space
        self["lr"] = lr
