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
from openrl.modules.networks.value_network import ValueNetwork
from openrl.modules.rl_module import RLModule
from openrl.modules.utils.util import update_linear_schedule


class DQNModule(RLModule):
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


        model_configs["q_net"] = ModelTrainConfig(
            lr=cfg.lr,
            model=(
                model_dict["q_net"]
                if model_dict and "q_net" in model_dict
                else ValueNetwork
            ),
            input_space=input_space,
        )
        model_configs["target_q_net"] = ModelTrainConfig(
            lr=cfg.lr,
            model=(
                model_dict["target_q_net"]
                if model_dict and "target_q_net" in model_dict
                else ValueNetwork
            ),
            input_space=input_space,
        )

        super(DQNModule, self).__init__(
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
        rnn_states_actor,
        masks,
        available_actions=None,
    ):

        values, actions, rnn_states_actor = self.models["q_net"](
            "original",
            obs,
            rnn_states_actor,
            masks,
            available_actions,
        )

        return values, actions, rnn_states_actor

    def get_values(self, critic_obs, rnn_states_critic, masks):
        values, _ = self.models["q_net"](critic_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        masks_batch=None,
    ):
        if masks_batch is None:
            masks_batch = masks

        values, _ = self.models["q_net"](
            obs, rnn_states, masks_batch, available_actions
        )

        return values

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None
    ):

        model = self.models["q_net"]

        actions, _, rnn_states_actor = model(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
        )

        return actions, rnn_states_actor

    def get_critic_value_normalizer(self):
        return self.models["q_net"].value_normalizer

    @staticmethod
    def init_rnn_states(
        rollout_num: int, agent_num: int, rnn_layers: int, hidden_size: int
    ):
        masks = np.ones((rollout_num * agent_num, 1), dtype=np.float32)
        rnn_state = np.zeros((rollout_num * agent_num, rnn_layers, hidden_size))
        return rnn_state, masks
