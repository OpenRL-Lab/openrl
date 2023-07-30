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

from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch

from openrl.modules.model_config import ModelTrainConfig
from openrl.modules.networks.ddpg_network import CriticNetwork
from openrl.modules.networks.sac_network import SACActorNetwork
from openrl.modules.rl_module import RLModule
from openrl.modules.utils.util import update_linear_schedule


class SACModule(RLModule):
    def __init__(
        self,
        cfg,
        input_space: gym.spaces.Box,
        act_space: gym.spaces.Box,
        device: Union[str, torch.device] = "cpu",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        model_dict: Optional[Dict[str, Any]] = None,
    ):
        model_configs = {}
        model_configs["actor"] = ModelTrainConfig(
            lr=cfg.actor_lr,
            model=(
                model_dict["actor"]
                if model_dict and "actor" in model_dict
                else SACActorNetwork
            ),
            input_space=input_space,
        )
        model_configs["critic"] = ModelTrainConfig(
            lr=cfg.critic_lr,
            model=(
                model_dict["critic"]
                if model_dict and "critic" in model_dict
                else CriticNetwork
            ),
            input_space=input_space,
        )
        model_configs["critic_target"] = ModelTrainConfig(
            lr=cfg.critic_lr,
            model=(
                model_dict["critic_target"]
                if model_dict and "critic_target" in model_dict
                else CriticNetwork
            ),
            input_space=input_space,
        )
        model_configs["critic_2"] = ModelTrainConfig(
            lr=cfg.critic_lr,
            model=(
                model_dict["critic_2"]
                if model_dict and "critic_2" in model_dict
                else CriticNetwork
            ),
            input_space=input_space,
        )
        model_configs["critic_target_2"] = ModelTrainConfig(
            lr=cfg.critic_lr,
            model=(
                model_dict["critic_target_2"]
                if model_dict and "critic_target_2" in model_dict
                else CriticNetwork
            ),
            input_space=input_space,
        )

        super().__init__(
            cfg=cfg,
            model_configs=model_configs,
            act_space=act_space,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        self.obs_space = input_space
        self.act_space = act_space
        self.cfg = cfg

        # alpha (can be dynamically adjusted)
        self.auto_alph = cfg.auto_alph
        if self.auto_alph:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=cfg.alpha_lr,
                eps=cfg.opti_eps,
                weight_decay=cfg.weight_decay,
            )
            self.optimizers["alpha"] = alpha_optimizer
            self.target_entropy = -np.prod(act_space.shape).item()
        else:
            self.log_alpha = torch.log(torch.tensor(cfg.alpha_value))

    def lr_decay(self, episode, episodes):
        update_linear_schedule(
            self.optimizers["critic"], episode, episodes, self.cfg.critic_lr
        )
        update_linear_schedule(
            self.optimizers["critic_2"], episode, episodes, self.cfg.critic_lr
        )
        update_linear_schedule(
            self.optimizers["actor"], episode, episodes, self.cfg.actor_lr
        )
        update_linear_schedule(
            self.optimizers["alpha"], episode, episodes, self.cfg.alpha_lr
        )

    def get_actions(self, obs, deterministic=True):
        actions, _ = self.models["actor"].evaluate(obs, deterministic=deterministic)

        return actions

    def get_values(self, obs, action, rnn_states_critic, masks):
        critic_values, _ = self.models["critic"](obs, action, rnn_states_critic, masks)

        return critic_values

    def evaluate_actor_loss(
        self,
        obs_batch,
        next_obs_batch,
        rnn_states_batch,
        rewards_batch,
        actions_batch,
        masks,
        action_masks=None,
        masks_batch=None,
    ):
        if masks_batch is None:
            masks_batch = masks

        action, log_prob = self.models["actor"].evaluate(obs_batch, deterministic=True)

        q_values = torch.min(
            self.models["critic"](obs_batch, action, rnn_states_batch, masks_batch)[0],
            self.models["critic_2"](obs_batch, action, rnn_states_batch, masks_batch)[
                0
            ],
        )

        actor_loss = (torch.exp(self.log_alpha) * log_prob - q_values).mean()

        return actor_loss, log_prob

    def get_q_values(
        self,
        obs_batch,
        next_obs_batch,
        rnn_states_batch,
        rewards_batch,
        actions_batch,
        masks,
        action_masks=None,
        masks_batch=None,
    ):
        if masks_batch is None:
            masks_batch = masks

        with torch.no_grad():
            next_action, next_log_prob = self.models["actor"].evaluate(
                next_obs_batch, deterministic=True
            )

            target_q_values, _ = self.models["critic"](
                next_obs_batch, next_action, rnn_states_batch, masks_batch
            )
            target_q_values = target_q_values.detach()
            target_q_values_2, _ = self.models["critic_2"](
                next_obs_batch, next_action, rnn_states_batch, masks_batch
            )
            target_q_values_2 = target_q_values_2.detach()

        current_q_values, _ = self.models["critic"](
            obs_batch, actions_batch, rnn_states_batch, masks_batch
        )

        current_q_values_2, _ = self.models["critic_2"](
            obs_batch, actions_batch, rnn_states_batch, masks_batch
        )

        return (
            target_q_values,
            target_q_values_2,
            current_q_values,
            current_q_values_2,
            next_log_prob,
        )

    def evaluate_actions(self):
        # This function is not required in SAC
        pass

    def act(self, obs, deterministic=True):
        actions, _ = self.models["actor"].evaluate(obs, deterministic=deterministic)

        return actions

    def get_critic_value_normalizer(self):
        return self.models["critic"].value_normalizer

    @staticmethod
    def init_rnn_states(
        rollout_num: int, agent_num: int, rnn_layers: int, hidden_size: int
    ):
        masks = np.ones((rollout_num * agent_num, 1), dtype=np.float32)
        rnn_state = np.zeros((rollout_num * agent_num, rnn_layers, hidden_size))
        return rnn_state, masks
