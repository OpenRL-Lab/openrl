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
from openrl.modules.networks.vdn_network import VDNNetwork
from openrl.modules.rl_module import RLModule
from openrl.modules.utils.util import update_linear_schedule


class VDNModule(RLModule):
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
        model_configs["vdn_net"] = ModelTrainConfig(
            lr=cfg.lr,
            model=(
                model_dict["vdn_net"]
                if model_dict and "vdn_net" in model_dict
                else VDNNetwork
            ),
            input_space=input_space,
        )
        model_configs["target_vdn_net"] = ModelTrainConfig(
            lr=cfg.lr,
            model=(
                model_dict["target_vdn_net"]
                if model_dict and "target_vdn_net" in model_dict
                else VDNNetwork
            ),
            input_space=input_space,
        )

        super(VDNModule, self).__init__(
            cfg=cfg,
            model_configs=model_configs,
            act_space=act_space,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        self.cfg = cfg

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizers["q_net"], episode, episodes, self.lr)

    def get_actions(
        self,
        obs,
        rnn_states,
        masks,
        action_masks=None,
    ):
        q_values, rnn_states = self.models["vdn_net"](
            "get_values",
            obs,
            rnn_states,
            masks,
            action_masks,
        )

        return q_values, rnn_states

    def get_values(self, obs, rnn_states_critic, masks):
        q_values, _ = self.models["vdn_net"](obs, rnn_states_critic, masks)
        return q_values

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
        critic_masks_batch=None,
    ):
        if masks_batch is None:
            masks_batch = masks

        q_tot = self.models["vdn_net"](
            "eval_actions",
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            action_masks,
        )

        max_next_q_tot = self.models["target_vdn_net"](
            "eval_actions_target",
            next_obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            action_masks,
        )

        return q_tot, max_next_q_tot

    def act(self, obs, rnn_states_actor, masks, action_masks=None):
        model = self.models["vdn_net"]

        q_values, rnn_states_actor = model(
            "eval_values",
            obs,
            rnn_states_actor,
            masks,
            action_masks,
        )

        return q_values, rnn_states_actor

    def get_critic_value_normalizer(self):
        return self.models["vdn_net"].value_normalizer

    @staticmethod
    def init_rnn_states(
        rollout_num: int, agent_num: int, rnn_layers: int, hidden_size: int
    ):
        masks = np.ones((rollout_num * agent_num, 1), dtype=np.float32)
        rnn_state = np.zeros((rollout_num * agent_num, rnn_layers, hidden_size))
        return rnn_state, masks
