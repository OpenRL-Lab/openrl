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

import numpy as np
import torch
from gymnasium import spaces
from rl_zoo3 import ALGOS
from torch import nn

from openrl.modules.utils.valuenorm import ValueNorm
from openrl.utils.util import check_v2 as check


class PolicyValueNetworkSB3(nn.Module):
    def __init__(
        self,
        cfg: Any,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        disable_drop_out: bool = True,
        extra_args=None,
    ):
        super(PolicyValueNetworkSB3, self).__init__()
        assert cfg.sb3_algo is not None
        assert cfg.sb3_model_path is not None
        self._use_valuenorm = cfg.use_valuenorm
        self.sb3_algo = cfg.sb3_algo
        model = ALGOS[cfg.sb3_algo].load(cfg.sb3_model_path, custom_objects={})

        self._policy_model = model.policy
        self.use_half = use_half
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.value_normalizer = (
            ValueNorm(1, device=device) if self._use_valuenorm else None
        )

    def get_actor_para(self):
        return self._policy_model.parameters()

    def get_critic_para(self):
        return self.get_actor_para()

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
        self, obs, rnn_states, masks, action_masks=None, deterministic=False
    ):
        if self.sb3_algo.endswith("_lstm"):
            return self.get_rnn_action(
                obs, rnn_states, masks, action_masks, deterministic
            )
        else:
            return self.get_naive_action(
                obs, rnn_states, masks, action_masks, deterministic
            )

    def get_rnn_action(
        self, obs, rnn_states, masks, action_masks=None, deterministic=False
    ):
        # actions, rnn_states = self._policy_model.predict(obs,rnn_states,deterministic=deterministic)
        #
        # rnn_states = check(rnn_states, self.use_half, self.tpdv)
        #
        # return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1), rnn_states
        raise NotImplementedError

    def get_naive_action(
        self, obs, rnn_states, masks, action_masks=None, deterministic=False
    ):
        observation = obs
        self._policy_model.set_training_mode(False)

        observation, vectorized_env = self._policy_model.obs_to_tensor(observation)

        with torch.no_grad():
            action_distribution = self._policy_model.get_distribution(observation)
            actions = action_distribution.get_actions(deterministic=deterministic)
            action_log_probs = action_distribution.log_prob(actions)
            # actions = self._policy_model._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = (
            actions.cpu().numpy().reshape((-1, *self._policy_model.action_space.shape))
        )

        if isinstance(self._policy_model.action_space, spaces.Box):
            if self.s_policy_model.quash_output:
                # Rescale to proper domain when using squashing

                actions = self._policy_model.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions,
                    self._policy_model.action_space.low,
                    self._policy_model.action_space.high,
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        actions = actions[:, np.newaxis]
        action_log_probs = action_log_probs[:, np.newaxis]
        return actions, action_log_probs, rnn_states

    def eval_actions(
        self, obs, rnn_states, action, masks, action_masks, active_masks=None
    ):
        obs = check(obs, self.use_half, self.tpdv)
        action = check(action, self.use_half, self.tpdv).squeeze()
        if self.sb3_algo.endswith("_lstm"):
            return self.eval_actions_rnn(
                obs, rnn_states, action, masks, action_masks, active_masks
            )
        else:
            return self.eval_actions_navie(
                obs, rnn_states, action, masks, action_masks, active_masks
            )

    def eval_actions_rnn(
        self, obs, rnn_states, action, masks, action_masks, active_masks
    ):
        values, log_prob, entropy = self._policy_model.evaluate_actions(
            obs, rnn_states, action
        )
        return log_prob, entropy.mean(), values

    def eval_actions_navie(
        self, obs, rnn_states, action, masks, action_masks, active_masks
    ):
        values, log_prob, entropy = self._policy_model.evaluate_actions(obs, action)
        return log_prob, entropy.mean(), values

    def get_values(self, obs, rnn_states, masks):
        if self.sb3_algo.endswith("_lstm"):
            return self.get_rnn_values(obs, rnn_states, masks)
        else:
            return self.get_naive_values(obs, rnn_states, masks)

    def get_rnn_values(self, obs, rnn_states, masks):
        raise NotImplementedError

    def get_naive_values(self, obs, rnn_states, masks):
        obs = check(obs, self.use_half, self.tpdv)
        values = self._policy_model.predict_values(obs)
        return values, rnn_states
