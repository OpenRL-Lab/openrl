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

import torch
import torch.nn as nn

from openrl.buffers.utils.util import get_policy_obs_space
from openrl.modules.networks.utils.act import ACTLayer
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mlp import MLPBase, MLPLayer
from openrl.modules.networks.utils.popart import PopArt
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        cfg,
        obs_space,
        critic_obs_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
    ):
        super(PolicyValueNetwork, self).__init__()
        self._gain = cfg.gain
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._recurrent_N = cfg.recurrent_N
        self._use_naive_recurrent_policy = cfg.use_naive_recurrent_policy
        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._concat_obs_as_critic_obs = cfg.concat_obs_as_critic_obs
        self._use_popart = cfg.use_popart
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.use_half = use_half
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        # obs space
        policy_obs_shape = get_policy_obs_space(obs_space)

        self.obs_prep = (
            CNNBase(cfg, policy_obs_shape)
            if len(policy_obs_shape) == 3
            else MLPBase(
                cfg,
                policy_obs_shape,
                use_attn_internal=cfg.use_attn_internal,
                use_cat_self=True,
            )
        )

        # critic_obs_shape = get_critic_obs_space(critic_obs_space)
        # self.critic_obs_prep = (
        #     CNNBase(cfg, critic_obs_shape)
        #     if len(critic_obs_shape) == 3
        #     else MLPBase(
        #         cfg,
        #         critic_obs_shape,
        #         use_attn_internal=True,
        #         use_cat_self=cfg.use_cat_self,
        #     )
        # )
        #
        self.critic_obs_prep = self.obs_prep

        # common layer
        self.common = MLPLayer(
            self.hidden_size,
            self.hidden_size,
            layer_N=0,
            use_orthogonal=self._use_orthogonal,
            activation_id=self._activation_id,
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        input_size = self.hidden_size

        # value
        if self._use_popart:
            self.v_out = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, 1))

        # action
        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )
        if use_half:
            self.half()
        self.to(self.device)

    def get_actions(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        obs = check(obs, self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.use_half, self.tpdv)

        x = obs
        x = self.obs_prep(x)

        # common
        actor_features = self.common(x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions, active_masks=None
    ):
        obs = check(obs, self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        action = check(action, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.use_half, self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks, self.use_half, self.tpdv)

        x = obs
        x = self.obs_prep(x)

        actor_features = self.common(x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions, active_masks
        )

        return action_log_probs, dist_entropy

    def get_values(self, critic_obs, rnn_states, masks):
        critic_obs = check(critic_obs, self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)

        share_x = critic_obs
        share_x = self.critic_obs_prep(share_x)

        critic_features = self.common(share_x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
