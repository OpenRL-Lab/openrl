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
from abc import ABC, abstractmethod

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseModule(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.models = {}
        self.optimizers = {}

    @abstractmethod
    def lr_decay(self, episode: int, episodes: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def restore(self, model_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, save_dir: str) -> None:
        raise NotImplementedError

    def convert_distributed_model(self) -> None:
        for model_name in self.models:
            model = self.models[model_name]
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.device], find_unused_parameters=True)
            self.models[model_name] = model
