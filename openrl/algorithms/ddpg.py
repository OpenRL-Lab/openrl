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
from openrl.modules.utils.util import get_grad_norm
from openrl.utils.util import check


class DDPGAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(cfg, init_module, agent_num, device)

        self.gamma = cfg.gamma
        self.tau = cfg.tau

    def prepare_critic_loss(
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
        import copy

        next_q_values, current_q_values = self.algo_module.evaluate_critic_loss(
            obs_batch,
            next_obs_batch,
            rnn_states_batch,
            rewards_batch,
            actions_batch,
            masks_batch,
            next_masks_batch,
            action_masks_batch,
            active_masks_batch,
        )
        with torch.no_grad():
            target_q_values = (
                rewards_batch
                + self.gamma * next_q_values * torch.tensor(next_masks_batch)
            ).detach()

        critic_loss = F.mse_loss(current_q_values, target_q_values)

        return critic_loss

    def prepare_actor_loss(
        self,
        obs_batch,
        next_obs_batch,
        rnn_states_batch,
        actions_batch,
        masks_batch,
        action_masks_batch,
        value_preds_batch,
        rewards_batch,
        active_masks_batch,
        turn_on,
    ):
        actor_loss = self.algo_module.evaluate_actor_loss(
            obs_batch,
            next_obs_batch,
            rnn_states_batch,
            rewards_batch,
            actions_batch,
            masks_batch,
            action_masks_batch,
            active_masks_batch,
        )

        return actor_loss

    def ddpg_update(self, sample, turn_on=True):
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

        # update critic network
        self.algo_module.optimizers["critic"].zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                critic_loss = self.prepare_critic_loss(
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
                critic_loss.backward()
        else:
            critic_loss = self.prepare_critic_loss(
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
            critic_loss.backward()

        if "transformer" in self.algo_module.models:
            raise NotImplementedError
        else:
            critic_para = self.algo_module.models["critic"].parameters()
            critic_grad_norm = get_grad_norm(critic_para)

        if self.use_amp:
            self.algo_module.scaler.unscale_(self.algo_module.optimizers["critic"])
            self.algo_module.scaler.step(self.algo_module.optimizers["critic"])
            self.algo_module.scaler.update()
        else:
            self.algo_module.optimizers["critic"].step()

        # update actor network
        self.algo_module.optimizers["actor"].zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                actor_loss = self.prepare_actor_loss(
                    obs_batch,
                    next_obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    action_masks_batch,
                    value_preds_batch,
                    rewards_batch,
                    active_masks_batch,
                    turn_on,
                )
                actor_loss.backward()
        else:
            actor_loss = self.prepare_actor_loss(
                obs_batch,
                next_obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                action_masks_batch,
                value_preds_batch,
                rewards_batch,
                active_masks_batch,
                turn_on,
            )
            actor_loss.backward()

        if "transformer" in self.algo_module.models:
            raise NotImplementedError
        else:
            actor_para = self.algo_module.models["actor"].parameters()
            actor_grad_norm = get_grad_norm(actor_para)

        if self.use_amp:
            self.algo_module.scaler.unscale_(self.algo_module.optimizers["actor"])
            self.algo_module.scaler.step(self.algo_module.optimizers["actor"])
            self.algo_module.scaler.update()
        else:
            self.algo_module.optimizers["actor"].step()

        # update target network
        for param, target_param in zip(
            self.algo_module.models["critic"].parameters(),
            self.algo_module.models["critic_target"].parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * param.data + self.tau * target_param.data
            )

        for param, target_param in zip(
            self.algo_module.models["actor"].parameters(),
            self.algo_module.models["actor_target"].parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * param.data + self.tau * target_param.data
            )

        # for others
        if self.world_size > 1:
            torch.cuda.synchronize()

        loss_list = []
        loss_list.append(critic_loss)
        loss_list.append(actor_loss)

        return loss_list

    def cal_value_loss(
        self,
        value_normalizer,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
    ):
        # TODO：to be finished
        raise NotImplementedError(
            "The calc_value_loss function in ddpg.py has not implemented yet"
        )

    def to_single_np(self, input):
        reshape_input = input.reshape(-1, self.agent_num, *input.shape[1:])
        return reshape_input[:, 0, ...]

    def train(self, buffer, turn_on=True):
        train_info = {}

        train_info["critic_loss"] = 0
        train_info["actor_loss"] = 0
        if self.world_size > 1:
            train_info["reduced_critic_loss"] = 0
            train_info["reduced_actor_loss"] = 0

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
                loss_list = self.ddpg_update(sample, turn_on)
                if self.world_size > 1:
                    train_info["reduced_critic_loss"] += reduce_tensor(
                        loss_list[0].data, self.world_size
                    )
                    train_info["reduced_actor_loss"] += reduce_tensor(
                        loss_list[1].data, self.world_size
                    )

                train_info["critic_loss"] += loss_list[0].item()
                train_info["actor_loss"] += loss_list[1].item()

        num_updates = 1 * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        for optimizer in self.algo_module.optimizers.values():
            if hasattr(optimizer, "sync_lookahead"):
                optimizer.sync_lookahead()

        return train_info
