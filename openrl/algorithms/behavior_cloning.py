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
import torch.nn as nn

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.modules.networks.utils.distributed_utils import reduce_tensor
from openrl.modules.utils.util import get_grad_norm
from openrl.utils.util import check


class BCAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self._use_share_model = cfg.use_share_model
        self.use_joint_action_loss = cfg.use_joint_action_loss
        super(BCAlgorithm, self).__init__(cfg, init_module, agent_num, device)
        self.train_list = [self.train_bc]

    def bc_update(self, sample, turn_on=True):
        for optimizer in self.algo_module.optimizers.values():
            optimizer.zero_grad()

        (
            critic_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            action_masks_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)

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
                    critic_obs_batch,
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch,
                    actions_batch,
                    masks_batch,
                    action_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    value_preds_batch,
                    return_batch,
                    active_masks_batch,
                    turn_on,
                )
            for loss in loss_list:
                self.algo_module.scaler.scale(loss).backward()
        else:
            loss_list, value_loss, policy_loss, dist_entropy, ratio = self.prepare_loss(
                critic_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                action_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                value_preds_batch,
                return_batch,
                active_masks_batch,
                turn_on,
            )
            for loss in loss_list:
                loss.backward()

        # else:

        actor_para = self.algo_module.models["policy"].parameters()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_para, self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(actor_para)

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
        critic_grad_norm = None
        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            ratio,
        )

    def to_single_np(self, input):
        reshape_input = input.reshape(-1, self.agent_num, *input.shape[1:])
        return reshape_input[:, 0, ...]

    def construct_loss_list(self, policy_loss, dist_entropy, value_loss, turn_on):
        loss_list = []
        if turn_on:
            final_p_loss = policy_loss - dist_entropy * self.entropy_coef
            loss_list.append(final_p_loss)
        return loss_list

    def prepare_loss(
        self,
        critic_obs_batch,
        obs_batch,
        rnn_states_batch,
        rnn_states_critic_batch,
        actions_batch,
        masks_batch,
        action_masks_batch,
        old_action_log_probs_batch,
        adv_targ,
        value_preds_batch,
        return_batch,
        active_masks_batch,
        turn_on,
    ):
        (
            values,
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.algo_module.evaluate_actions(
            critic_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            action_masks_batch,
            active_masks_batch,
            critic_masks_batch=None,
        )

        if self.use_joint_action_loss:
            action_log_probs_copy = (
                action_log_probs.reshape(-1, self.agent_num, action_log_probs.shape[-1])
                .sum(dim=(1, -1), keepdim=True)
                .reshape(-1, 1)
            )
            policy_loss = -action_log_probs_copy.mean()
        else:
            policy_loss = -action_log_probs.mean()

        value_loss = None
        ratio = None

        loss_list = self.construct_loss_list(
            policy_loss, dist_entropy, value_loss, turn_on
        )
        return loss_list, value_loss, policy_loss, dist_entropy, ratio

    def get_data_generator(self, buffer):
        advantages = None
        if self._use_recurrent_policy:
            if self.use_joint_action_loss:
                data_generator = buffer.recurrent_generator_v3(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            else:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(
                advantages, self.num_mini_batch
            )
        else:
            data_generator = buffer.feed_forward_generator(
                advantages, self.num_mini_batch
            )
        return data_generator

    def train_bc(self, buffer, turn_on):
        train_info = {}

        train_info["policy_loss"] = 0

        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0

        if self.world_size > 1:
            train_info["reduced_value_loss"] = 0
            train_info["reduced_policy_loss"] = 0

        for _ in range(self.bc_epoch):
            data_generator = self.get_data_generator(buffer)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                ) = self.bc_update(sample, turn_on)

                if self.world_size > 1:
                    train_info["reduced_policy_loss"] += reduce_tensor(
                        policy_loss.data, self.world_size
                    )

                train_info["policy_loss"] += policy_loss.item()

                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm

        num_updates = self.bc_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def train(self, buffer, turn_on=True):
        train_info = {}
        for train_func in self.train_list:
            train_info.update(train_func(buffer, turn_on))

        for optimizer in self.algo_module.optimizers.values():
            if hasattr(optimizer, "sync_lookahead"):
                optimizer.sync_lookahead()

        return train_info
