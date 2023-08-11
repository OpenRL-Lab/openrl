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
import torch.nn.functional as F

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.modules.networks.utils.distributed_utils import reduce_tensor
from openrl.modules.utils.util import get_grad_norm, huber_loss, mse_loss
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
        self.update_count = 0
        self.target_update_frequency = cfg.train_interval

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
            next_masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            action_masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        next_masks_batch = check(next_masks_batch).to(**self.tpdv)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss_list = self.prepare_loss(
                    obs_batch,
                    next_obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    next_masks_batch,
                    action_masks_batch,
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
                next_masks_batch,
                action_masks_batch,
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

        if self.update_count % self.target_update_frequency == 0:
            self.update_count = 0
            self.algo_module.models["target_q_net"].load_state_dict(
                self.algo_module.models["q_net"].state_dict()
            )
        else:
            self.update_count += 1
        return loss

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
        next_masks_batch,
        action_masks_batch,
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
            next_masks_batch,
            action_masks_batch,
            active_masks_batch,
            critic_masks_batch=critic_masks_batch,
        )

        q_targets = rewards_batch + self.gamma * max_next_q_values * next_masks_batch
        q_loss = torch.mean(F.mse_loss(q_values, q_targets.detach()))  # 均方误差损失函数

        loss_list.append(q_loss)

        return loss_list

    def train(self, buffer, turn_on=True):
        train_info = {}

        train_info["q_loss"] = 0

        if self.world_size > 1:
            train_info["reduced_q_loss"] = 0

        # todo add rnn and transformer
        for _ in range(self.num_mini_batch):
            if "transformer" in self.algo_module.models:
                raise NotImplementedError
            elif self._use_recurrent_policy:
                raise NotImplementedError
            elif self._use_naive_recurrent:
                raise NotImplementedError
            else:
                data_generator = buffer.feed_forward_generator(
                    None,
                    num_mini_batch=self.num_mini_batch,
                    mini_batch_size=self.mini_batch_size,
                )

            for sample in data_generator:
                (q_loss) = self.dqn_update(sample, turn_on)
                # print(q_loss)
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
