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
from torch.nn.parallel import DistributedDataParallel

from openrl.algorithms.ppo import PPOAlgorithm


class A2CAlgorithm(PPOAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super(A2CAlgorithm, self).__init__(cfg, init_module, agent_num, device)

        self.num_mini_batch = 1

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
        if self.use_joint_action_loss:
            critic_obs_batch = self.to_single_np(critic_obs_batch)
            rnn_states_critic_batch = self.to_single_np(rnn_states_critic_batch)
            critic_masks_batch = self.to_single_np(masks_batch)
            value_preds_batch = self.to_single_np(value_preds_batch)
            return_batch = self.to_single_np(return_batch)
            adv_targ = adv_targ.reshape(-1, self.agent_num, 1)
            adv_targ = adv_targ[:, 0, :]

        else:
            critic_masks_batch = masks_batch

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
            critic_masks_batch=critic_masks_batch,
        )

        if self.use_joint_action_loss:
            active_masks_batch = active_masks_batch.reshape(-1, self.agent_num, 1)
            active_masks_batch = active_masks_batch[:, 0, :]

        policy_gradient_loss = -adv_targ.detach() * action_log_probs
        if self._use_policy_active_masks:
            policy_action_loss = (
                torch.sum(policy_gradient_loss, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = torch.sum(
                policy_gradient_loss, dim=-1, keepdim=True
            ).mean()

        if self._use_policy_vhead:
            if isinstance(self.algo_module.models["actor"], DistributedDataParallel):
                policy_value_normalizer = self.algo_module.models[
                    "actor"
                ].module.value_normalizer
            else:
                policy_value_normalizer = self.algo_module.models[
                    "actor"
                ].value_normalizer
            policy_value_loss = self.cal_value_loss(
                policy_value_normalizer,
                policy_values,
                value_preds_batch,
                return_batch,
                active_masks_batch,
            )
            policy_loss = (
                policy_action_loss + policy_value_loss * self.policy_value_loss_coef
            )
        else:
            policy_loss = policy_action_loss

        # critic update
        if self._use_share_model:
            value_normalizer = self.algo_module.models["model"].value_normalizer
        elif isinstance(self.algo_module.models["critic"], DistributedDataParallel):
            value_normalizer = self.algo_module.models["critic"].module.value_normalizer
        else:
            value_normalizer = self.algo_module.get_critic_value_normalizer()
        value_loss = self.cal_value_loss(
            value_normalizer,
            values,
            value_preds_batch,
            return_batch,
            active_masks_batch,
        )

        loss_list = self.construct_loss_list(
            policy_loss, dist_entropy, value_loss, turn_on
        )
        ratio = np.zeros(1)
        return loss_list, value_loss, policy_loss, dist_entropy, ratio

    def train(self, buffer, turn_on: bool = True):
        train_info = super(A2CAlgorithm, self).train(buffer, turn_on)
        train_info.pop("ratio", None)
        return train_info
