#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The OpenRL Authors.
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
from typing import Any

import torch

from openrl.modules.networks.utils.nlp.causal_policy import CausalLMActorCriticPolicy
from openrl.utils.util import check_v2 as check


class PolicyValueNetworkGPT(CausalLMActorCriticPolicy):
    def __init__(
        self,
        cfg: Any,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        disable_drop_out: bool = True,
    ):
        self.disable_drop_out = disable_drop_out
        self._use_valuenorm = cfg.use_valuenorm
        super(CausalLMActorCriticPolicy, self).__init__(
            input_space,
            action_space,
            model_name=cfg.model_path,
            device=device,
        )
        self.use_half = use_half
        self.tpdv = dict(dtype=torch.float32, device=device)

    def get_actor_para(self):
        return self._policy_model.parameters()

    def get_critic_para(self):
        return self._value_model.parameters()

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.get_actions(*args, **kwargs)
        elif forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        elif forward_type == "get_values":
            return self.get_values(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_actions(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)

        past_model_kwargs = None
        policy_output, past_model_kwargs = super().get_distribution(
            obs, past_model_kwargs
        )
        actions = policy_output.mode() if deterministic else policy_output.sample()
        action_log_probs = policy_output.log_prob(actions)

        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1), rnn_states
        # TODO: add past_model_kwargs, i.e., past key value.

    def eval_actions(
        self, obs, rnn_states, action, masks, available_actions, active_masks=None
    ):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        action = check(action, self.use_half, self.tpdv).squeeze()

        eval_output = super().evaluate_actions(obs, action)
        action_log_probs = eval_output.log_prob
        dist_entropy = eval_output.entropy
        values = eval_output.values

        return action_log_probs.unsqueeze(-1), dist_entropy.mean(), values

    def get_values(self, obs, rnn_states, masks):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)

        value_output = super().forward_value(obs)
        values = value_output.values

        return values, rnn_states

    def get_log_probs_ref_model(self, obs, action):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        action = check(action, self.use_half, self.tpdv)
        action = action.squeeze(-1)

        policy_output = super().get_log_probs_ref_model(obs, action)
        action_log_probs = policy_output.log_probs

        return action_log_probs.detach().cpu().numpy()
