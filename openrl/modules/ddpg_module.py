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
from openrl.modules.networks.ddpg_network import ActorNetwork, CriticNetwork
from openrl.modules.rl_module import RLModule
from openrl.modules.utils.util import update_linear_schedule


class DDPGModule(RLModule):
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
                else ActorNetwork
            ),
            input_space=input_space,
        )
        model_configs["actor_target"] = ModelTrainConfig(
            lr=cfg.actor_lr,
            model=(
                model_dict["actor_target"]
                if model_dict and "actor_target" in model_dict
                else ActorNetwork
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

    def lr_decay(self, episode, episodes):
        update_linear_schedule(
            self.optimizers["critic"], episode, episodes, self.cfg.critic_lr
        )
        update_linear_schedule(
            self.optimizers["actor"], episode, episodes, self.cfg.actor_lr
        )

    def get_actions(
        self,
        obs,
        # rnn_states,
        # masks,
        # action_masks=None,
    ):
        action = self.models["actor"](obs)

        return action

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
        actions = self.get_actions(obs_batch)

        actor_loss, _ = self.models["critic"](
            obs_batch, actions, rnn_states_batch, masks_batch
        )
        actor_loss = -actor_loss.mean()

        return actor_loss

    def evaluate_critic_loss(
        self,
        obs_batch,
        next_obs_batch,
        rnn_states_batch,
        rewards_batch,
        actions_batch,
        masks,
        next_masks_batch,
        action_masks=None,
        masks_batch=None,
    ):
        if masks_batch is None:
            masks_batch = masks
        with torch.no_grad():
            next_q_values, _ = self.models["critic_target"](
                next_obs_batch,
                self.models["actor_target"](next_obs_batch),
                rnn_states_batch,
                masks_batch,
            )
        current_q_values, _ = self.models["critic"](
            obs_batch, actions_batch, rnn_states_batch, masks_batch
        )

        return next_q_values, current_q_values

    def evaluate_actions(
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
        print("在ddpg_module中调用了evaluate_actions函数，该函数未实现")

    def act(
        self,
        obs,
        # rnn_states_actor,
        # masks,
        # action_masks=None
        deterministic: bool,
    ):
        action = self.models["actor"](obs)

        return action

    def get_critic_value_normalizer(self):
        return self.models["critic"].value_normalizer

    @staticmethod
    def init_rnn_states(
        rollout_num: int, agent_num: int, rnn_layers: int, hidden_size: int
    ):
        masks = np.ones((rollout_num * agent_num, 1), dtype=np.float32)
        rnn_state = np.zeros((rollout_num * agent_num, rnn_layers, hidden_size))
        return rnn_state, masks
