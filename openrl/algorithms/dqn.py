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


class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super(DQNAlgorithm, self).__init__(cfg, init_module, agent_num, device)

        self.gamma = cfg.gamma

    def dqn_update(self, sample, turn_on=True):
        for optimizer in self.algo_module.optimizers.values():
            optimizer.zero_grad()

        (
            obs_batch,
            _,
            next_obs_batch,
            _,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            rewards_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss_list = self.prepare_loss(
                    obs_batch,
                    next_obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    available_actions_batch,
                    value_preds_batch,
                    rewards_batch,
                    active_masks_batch,
                    turn_on,
                )
            for loss in loss_list:
                self.algo_module.scaler.scale(loss).backward()
        else:
            loss_list = self.prepare_loss(
                obs_batch,
                next_obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                value_preds_batch,
                rewards_batch,
                active_masks_batch,
                turn_on,
            )
            for loss in loss_list:
                loss.backward()

        if "transformer" in self.algo_module.models:
            raise NotImplementedError
        else:
            actor_para = self.algo_module.models["q_net"].parameters()
            actor_grad_norm = get_gard_norm(actor_para)

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

        return loss

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
        next_obs_batch,
        rnn_states_batch,
        actions_batch,
        masks_batch,
        available_actions_batch,
        value_preds_batch,
        rewards_batch,
        active_masks_batch,
        turn_on,
    ):
        loss_list = []
        critic_masks_batch = masks_batch

        (q_values, max_next_q_values) = self.algo_module.evaluate_actions(
            obs_batch,
            next_obs_batch,
            rnn_states_batch,
            rewards_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            critic_masks_batch=critic_masks_batch,
        )

        q_targets = rewards_batch + self.gamma * max_next_q_values
        q_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        loss_list.append(q_loss)
        return loss_list

    def train(self, buffer, turn_on=True):
        train_info = {}

        train_info["q_loss"] = 0

        if self.world_size > 1:
            train_info["reduced_q_loss"] = 0

        # todo add rnn and transformer
        # update once
        for _ in range(1):
            if "transformer" in self.algo_module.models:
                raise NotImplementedError
            elif self._use_recurrent_policy:
                raise NotImplementedError
            elif self._use_naive_recurrent:
                raise NotImplementedError
            else:
                data_generator = buffer.feed_forward_generator(_, self.num_mini_batch)

            for sample in data_generator:
                (
                    q_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                ) = self.dqn_update(sample, turn_on)

                if self.world_size > 1:
                    train_info["reduced_q_loss"] += reduce_tensor(
                        q_loss.data, self.world_size
                    )

                train_info["q_loss"] += q_loss.item()

        num_updates = 1 * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        for optimizer in self.algo_module.optimizers.values():
            if hasattr(optimizer, "sync_lookahead"):
                optimizer.sync_lookahead()

        return train_info
