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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from openrl.modules.networks.utils.mlp import MLPLayer
from openrl.modules.networks.utils.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device,
        use_half,
        extra_args=None,
    ):
        super(Discriminator, self).__init__()
        hidden_dim = cfg.gail_hidden_size
        layer_num = cfg.gail_layer_num
        self.cfg = cfg
        self.device = device
        self.critic_obs_process_func = (
            extra_args["critic_obs_process_func"]
            if extra_args is not None and "critic_obs_process_func" in extra_args
            else lambda _: _
        )

        self.base = MLPLayer(
            input_space,
            hidden_dim,
            layer_N=layer_num,
            use_orthogonal=cfg.use_orthogonal,
            activation_id=cfg.activation_id,
        )

        self.gail_out = nn.Linear(hidden_dim, 1)
        self.gail_out.weight.data.mul_(0.1)
        self.gail_out.bias.data.mul_(0.0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.gail_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.first_train = True
        self.to(device)
        self.train()

    def compute_grad_pen(
        self, expert_state, expert_action, policy_state, policy_action, lambda_=10
    ):
        alpha = torch.rand(expert_state.size(0), 1)

        if self.cfg.gail_use_action:
            expert_data = torch.cat([expert_state, expert_action], dim=-1)
            policy_data = torch.cat([policy_state, policy_action], dim=-1)
        else:
            expert_data = expert_state
            policy_data = policy_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.gail_out(self.base(mixup_data))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, buffer, obsfilt=None):
        self.train()

        policy_data_generator = buffer.feed_forward_critic_obs_generator(
            None,
            mini_batch_size=expert_loader.batch_size,
            critic_obs_process_func=self.critic_obs_process_func,
        )

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[4]
            policy_state = torch.from_numpy(policy_state).to(self.device)

            if self.cfg.gail_use_action:
                policy_action = torch.from_numpy(policy_action).to(self.device)

            if self.cfg.gail_use_action:
                policy_d = self.gail_out(
                    self.base(torch.cat([policy_state, policy_action], dim=-1))
                )

            else:
                policy_d = self.gail_out(self.base(policy_state))

            expert_state, expert_action = expert_batch
            expert_state = expert_state.reshape(-1, *expert_state.shape[2:])
            expert_action = expert_action.reshape(-1, *expert_action.shape[2:])

            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
                expert_state = torch.FloatTensor(expert_state).to(self.device)
            else:
                expert_state = expert_state.to(self.device)

            if self.cfg.gail_use_action:
                expert_action = expert_action.to(self.device)

                expert_d = self.gail_out(
                    self.base(torch.cat([expert_state, expert_action], dim=-1))
                )
            else:
                expert_d = self.gail_out(self.base(expert_state))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d, torch.zeros(expert_d.size()).to(self.device)
            )
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d, torch.ones(policy_d.size()).to(self.device)
            )

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(
                expert_state, expert_action, policy_state, policy_action
            )

            if not self.first_train:
                loss += (gail_loss + grad_pen).item()
                n += 1
            else:
                self.first_train = False

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n if n > 0 else 0

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            state_shape = state.shape
            masks_shape = masks.shape
            state = self.critic_obs_process_func(state.reshape(-1, state_shape[-1]))
            state = torch.from_numpy(state).to(self.device)

            masks = torch.from_numpy(masks).to(self.device).reshape(-1, masks_shape[-1])
            if self.cfg.gail_use_action:
                action_shape = action.shape
                action = (
                    torch.from_numpy(action)
                    .to(self.device)
                    .reshape(-1, action_shape[-1])
                )
                d = self.gail_out(self.base(torch.cat([state, action], dim=-1)))
            else:
                d = self.gail_out(self.base(state))

            s = torch.sigmoid(d) + 1e-8
            reward = -s.log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                reward = reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

            reward = reward.reshape((*state_shape[:2], reward.shape[-1])).cpu().numpy()

            return reward
