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

from openrl.buffers.utils.util import get_policy_obs, get_policy_obs_space
from openrl.modules.networks.base_policy_network import BasePolicyNetwork
from openrl.modules.networks.utils.act import ACTLayer
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase, MLPLayer
from openrl.modules.networks.utils.popart import PopArt
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class PolicyNetwork(BasePolicyNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
    ) -> None:
        super(PolicyNetwork, self).__init__(cfg, device)
        self.hidden_size = cfg.hidden_size

        self._gain = cfg.gain
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_policy_active_masks = cfg.use_policy_active_masks
        self._use_naive_recurrent_policy = cfg.use_naive_recurrent_policy
        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_influence_policy = cfg.use_influence_policy
        self._influence_layer_N = cfg.influence_layer_N
        self._use_policy_vhead = cfg.use_policy_vhead
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

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                policy_obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        if self._use_policy_vhead:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
                self._use_orthogonal
            ]

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))

            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))
        if use_half:
            self.half()
        self.to(device)

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.forward_original(*args, **kwargs)
        elif forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        else:
            raise NotImplementedError

    def forward_original(
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
            # if self.use_half:
            #     obs = obs.half()
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions, self.use_half, self.tpdv)

        actor_features = self.base(policy_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(policy_obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states

    def eval_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key], self.use_half, self.tpdv)
        else:
            obs = check(obs, self.use_half, self.tpdv)

        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        action = check(action, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions, self.use_half, self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks, self.use_half, self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key], self.use_half, self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        values = self.v_out(actor_features)

        return values
