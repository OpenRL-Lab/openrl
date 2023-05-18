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

from openrl.buffers.utils.util import get_policy_obs, get_policy_obs_space
from openrl.modules.networks.base_policy_network import BasePolicyNetwork
from openrl.modules.networks.utils.act import ACTLayer
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.utils.util import check_v2 as check


class QNetwork(BasePolicyNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
    ) -> None:
        super(QNetwork, self).__init__(cfg, device)
        self.hidden_size = cfg.hidden_size

        self._gain = cfg.gain
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_policy_active_masks = cfg.use_policy_active_masks
        self._use_naive_recurrent_policy = cfg.use_naive_recurrent_policy
        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._recurrent_N = cfg.recurrent_N
        self.use_half = use_half
        self.tpdv = dict(dtype=torch.float32, device=device)

        policy_obs_shape = get_policy_obs_space(input_space)

        if "Dict" in policy_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(
                cfg, policy_obs_shape, cnn_layers_params=cfg.cnn_layers_params
            )
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(cfg, policy_obs_shape)
                if len(policy_obs_shape) == 3
                else MLPBase(
                    cfg,
                    policy_obs_shape,
                    use_attn_internal=cfg.use_attn_internal,
                    use_cat_self=True,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                rnn_type=cfg.rnn_type,
            )
            input_size = self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        if use_half:
            self.half()
        self.to(device)

    def forward(
        self, raw_obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        policy_obs = get_policy_obs(raw_obs)
        if self._mixed_obs:
            for key in policy_obs.keys():
                policy_obs[key] = check(policy_obs[key], self.use_half, self.tpdv)
                if self.use_half:
                    policy_obs[key].half()
        else:
            policy_obs = check(policy_obs, self.use_half, self.tpdv)

        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions, self.use_half, self.tpdv)

        actor_features = self.base(policy_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states
