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


class BaseAlgorithm(ABC):
    def __init__(self, cfg, init_module, agent_num: int, device=torch.device("cpu")):
        self.cfg = cfg

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.algo_module = init_module

        self.world_size = self.algo_module.world_size or 1
        self.clip_param = cfg.clip_param
        self.ppo_epoch = cfg.ppo_epoch
        self.num_mini_batch = cfg.num_mini_batch
        self.data_chunk_length = cfg.data_chunk_length
        self.policy_value_loss_coef = cfg.policy_value_loss_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.huber_delta = cfg.huber_delta

        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_naive_recurrent = cfg.use_naive_recurrent_policy
        self._use_max_grad_norm = cfg.use_max_grad_norm
        self._use_clipped_value_loss = cfg.use_clipped_value_loss
        self._use_huber_loss = cfg.use_huber_loss
        self._use_popart = cfg.use_popart
        self._use_valuenorm = cfg.use_valuenorm
        self._use_value_active_masks = cfg.use_value_active_masks
        self._use_policy_active_masks = cfg.use_policy_active_masks
        self._use_policy_vhead = cfg.use_policy_vhead

        self.agent_num = agent_num

        self._use_adv_normalize = cfg.use_adv_normalize

        # for tranformer
        self.dec_actor = cfg.dec_actor

        self.use_amp = cfg.use_amp

        self.dual_clip_ppo = cfg.dual_clip_ppo
        self.dual_clip_coeff = torch.tensor(cfg.dual_clip_coeff).to(self.device)

        assert not (
            self._use_popart and self._use_valuenorm
        ), "self._use_popart and self._use_valuenorm can not be set True simultaneously"

    @abstractmethod
    def train(self, buffer, turn_on=True):
        raise NotImplementedError

    def prep_training(self):
        for model in self.algo_module.models.values():
            model.train()

    def prep_rollout(self):
        for model in self.algo_module.models.values():
            model.eval()
