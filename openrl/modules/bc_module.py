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
from openrl.modules.networks.policy_network import PolicyNetwork
from openrl.modules.rl_module import RLModule
from openrl.modules.utils.util import update_linear_schedule


class BCModule(RLModule):
    def __init__(
        self,
        cfg,
        policy_input_space: gym.spaces.Box,
        critic_input_space: gym.spaces.Box,
        act_space: gym.spaces.Box,
        share_model: bool = False,
        device: Union[str, torch.device] = "cpu",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        model_dict: Optional[Dict[str, Any]] = None,
    ):
        self.share_model = share_model
        self.policy_input_space = policy_input_space
        self.critic_input_space = critic_input_space
        self.model_dict = model_dict

        super(BCModule, self).__init__(
            cfg=cfg,
            act_space=act_space,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    def get_model_configs(self, cfg) -> Dict[str, Any]:
        model_configs = {
            "policy": ModelTrainConfig(
                lr=cfg.lr,
                model=(
                    self.model_dict["policy"]
                    if self.model_dict and "policy" in self.model_dict
                    else PolicyNetwork
                ),
                input_space=self.policy_input_space,
            )
        }

        return model_configs

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizers["policy"], episode, episodes, self.lr)

    def get_actions(
        self,
        critic_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        action_masks=None,
        deterministic=False,
    ):
        actions, action_log_probs, rnn_states_actor = self.models["policy"](
            "original",
            obs,
            rnn_states_actor,
            masks,
            action_masks,
            deterministic,
        )

        values, rnn_states_critic = None, None
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, critic_obs, rnn_states_critic, masks):
        return None

    def evaluate_actions(
        self,
        critic_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        action_masks=None,
        active_masks=None,
        critic_masks_batch=None,
    ):
        values = None

        action_log_probs, dist_entropy, policy_values = self.models["policy"](
            "eval_actions",
            obs,
            rnn_states_actor,
            action,
            masks,
            action_masks,
            active_masks,
        )

        return values, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, action_masks=None, deterministic=False):
        model = self.models["policy"]

        actions, _, rnn_states_actor = model(
            "original",
            obs,
            rnn_states_actor,
            masks,
            action_masks,
            deterministic,
        )

        return actions, rnn_states_actor

    def get_critic_value_normalizer(self):
        return None

    @staticmethod
    def init_rnn_states(
        rollout_num: int, agent_num: int, rnn_layers: int, hidden_size: int
    ):
        masks = np.ones((rollout_num * agent_num, 1), dtype=np.float32)
        rnn_state = np.zeros((rollout_num * agent_num, rnn_layers, hidden_size))
        return rnn_state, masks
