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
from torch.nn.parallel import DistributedDataParallel

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.modules.networks.utils.distributed_utils import reduce_tensor
from openrl.modules.utils.util import get_gard_norm, huber_loss, mse_loss
from openrl.utils.util import check


class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self._use_share_model = cfg.use_share_model
        self.use_joint_action_loss = cfg.use_joint_action_loss
        super(DQNAlgorithm, self).__init__(cfg, init_module, agent_num, device)

    def dqn_update(self, sample, turn_on=True):
        for optimizer in self.algo_module.optimizers.values():
            optimizer.zero_grad()

        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            available_actions_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                (
                    loss_list,
                    value_loss,
                    policy_loss,
                    dist_entropy,
                    ratio,
                ) = self.prepare_loss(
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    available_actions_batch,
                    value_preds_batch,
                    return_batch,
                    active_masks_batch,
                    turn_on,
                )
            for loss in loss_list:
                self.algo_module.scaler.scale(loss).backward()
        else:
            loss_list, value_loss, policy_loss, dist_entropy, ratio = self.prepare_loss(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                value_preds_batch,
                return_batch,
                active_masks_batch,
                turn_on,
            )
            for loss in loss_list:
                loss.backward()

        if "transformer" in self.algo_module.models:
            if self._use_max_grad_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.algo_module.models["transformer"].parameters(),
                    self.max_grad_norm,
                )
            else:
                grad_norm = get_gard_norm(
                    self.algo_module.models["transformer"].parameters()
                )
            critic_grad_norm = grad_norm
            actor_grad_norm = grad_norm

        else:
            if self._use_share_model:
                actor_para = self.algo_module.models["model"].get_actor_para()
            else:
                actor_para = self.algo_module.models["policy"].parameters()

            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    actor_para, self.max_grad_norm
                )
            else:
                actor_grad_norm = get_gard_norm(actor_para)

            if self._use_share_model:
                critic_para = self.algo_module.models["model"].get_critic_para()
            else:
                critic_para = self.algo_module.models["critic"].parameters()

            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    critic_para, self.max_grad_norm
                )
            else:
                critic_grad_norm = get_gard_norm(critic_para)

        if self.use_amp:
            for optimizer in self.algo_module.optimizers.values():
                self.algo_module.scaler.unscale_(optimizer)

            for optimizer in self.algo_module.optimizers.values():
                self.algo_module.scaler.step(optimizer)

            self.algo_module.scaler.update()
        else:
            for optimizer in self.algo_module.optimizers.values():
                optimizer.step()

        if self.world_size > 1:
            torch.cuda.synchronize()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            ratio,
        )

    def cal_value_loss(
        self,
        value_normalizer,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def to_single_np(self, input):
        reshape_input = input.reshape(-1, self.agent_num, *input.shape[1:])
        return reshape_input[:, 0, ...]

    def prepare_loss(
        self,
        obs_batch,
        rnn_states_batch,
        actions_batch,
        masks_batch,
        available_actions_batch,
        value_preds_batch,
        return_batch,
        active_masks_batch,
        turn_on,
    ):
        raise NotImplementedError

    def train(self, buffer, turn_on=True):
        raise NotImplementedError
