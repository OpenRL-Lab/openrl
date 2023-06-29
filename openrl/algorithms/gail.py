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

import torch

from openrl.algorithms.ppo import PPOAlgorithm
from openrl.datasets.expert_dataset import ExpertDataset


class GAILAlgorithm(PPOAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super(GAILAlgorithm, self).__init__(cfg, init_module, agent_num, device)
        self.train_list.append(self.train_gail)
        self.gail_epoch = cfg.gail_epoch
        assert cfg.expert_data is not None
        expert_dataset = ExpertDataset(file_name=cfg.expert_data)
        drop_last = len(expert_dataset) > self.cfg.gail_batch_size
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=self.cfg.gail_batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

    def train_gail(self, buffer, turn_on):
        train_info = {"gail_loss": 0}

        for _ in range(self.gail_epoch):
            loss = self.algo_module.models["gail_discriminator"].update(
                self.dataset_loader, buffer
            )
            train_info["gail_loss"] += loss

        for k in train_info.keys():
            train_info[k] /= self.gail_epoch
        return train_info
