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

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from openrl.buffers.utils.util import get_critic_obs_space, get_policy_obs_space
from openrl.modules.networks.base_policy_network import BasePolicyNetwork
from openrl.modules.networks.base_value_network import BaseValueNetwork
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class ActorNetwork(BasePolicyNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        extra_args=None,
    ) -> None:
        super().__init__(cfg, device)
        self.hidden_size = cfg.hidden_size
        self.action_space = action_space

        self.use_half = use_half
        self._use_orthogonal = cfg.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        obs_shape = get_policy_obs_space(input_space)

        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(cfg, obs_shape, cnn_layers_params=cfg.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(cfg, obs_shape)
                if len(obs_shape) == 3
                else MLPBase(
                    cfg,
                    obs_shape,
                    use_attn_internal=cfg.use_attn_internal,
                    use_cat_self=True,
                )
            )

        input_size = self.base.output_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            self.actor_out = init_(nn.Linear(input_size, action_space.n))
        elif isinstance(self.action_space, gym.spaces.box.Box):
            self.actor_out = init_(nn.Linear(input_size, action_space.shape[0]))
        else:
            raise NotImplementedError("This type of game has not been implemented.")

        if use_half:
            self.half()
        self.to(device)

    def forward(self, obs):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        features = self.base(obs)

        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            features = F.relu(features)
            action = self.actor_out(features)
        elif isinstance(self.action_space, gym.spaces.box.Box):
            action = self.actor_out(features)
            action = F.tanh(action)
            action = (action + 1) / 2 * (
                torch.tensor(self.action_space.high)
                - torch.tensor(self.action_space.low)
            ) + torch.tensor(self.action_space.low)

        else:
            raise NotImplementedError("This type of game has not been implemented.")

        return action


class CriticNetwork(BaseValueNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        extra_args=None,
    ) -> None:
        super().__init__(cfg, device)
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

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        state_shape = get_critic_obs_space(input_space)[0]
        action_shape = get_policy_obs_space(action_space)[0]
        input_shape = (state_shape + action_shape,)

        if "Dict" in input_shape.__class__.__name__:
            self._mixed_obs = True
            self.input_base = MIXBase(
                cfg, input_shape, cnn_layers_params=cfg.cnn_layers_params
            )
        else:
            self._mixed_obs = False
            self.input_base = MLPBase(
                cfg,
                input_shape,
                use_attn_internal=cfg.use_attn_internal,
                use_cat_self=True,
            )

        input_size = self.input_base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                rnn_type=cfg.rnn_type,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.critic_out = init_(nn.Linear(input_size, 1))

        if use_half:
            self.half()
        self.to(device)

    def forward(self, state, action, rnn_states, masks):
        if self._mixed_obs:
            for key in state.keys():
                state[key] = check(state[key]).to(**self.tpdv)
            for key in action.keys():
                action[key] = check(action[key]).to(**self.tpdv)
        else:
            state = check(state).to(**self.tpdv)
            action = check(action).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        input = torch.cat((state, action), 1)
        # features = F.relu(self.input_base(input))
        features = self.input_base(input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        critic_out = self.critic_out(features)

        return critic_out, rnn_states


class CriticNetwork_v0(BaseValueNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
    ) -> None:
        super().__init__(cfg, device)
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

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        state_shape = get_critic_obs_space(input_space)
        action_shape = get_policy_obs_space(action_space)

        if "Dict" in state_shape.__class__.__name__:
            self._mixed_obs = True
            self.state_base = MIXBase(
                cfg, state_shape, cnn_layers_params=cfg.cnn_layers_params
            )
        else:
            self._mixed_obs = False
            self.state_base = (
                CNNBase(cfg, state_shape)
                if len(state_shape) == 3
                else MLPBase(
                    cfg,
                    state_shape,
                    use_attn_internal=cfg.use_attn_internal,
                    use_cat_self=True,
                )
            )

        if "Dict" in action_shape.__class__.__name__:
            self._mixed_obs = True
            self.action_base = MIXBase(
                cfg, action_shape, cnn_layers_params=cfg.cnn_layers_params
            )
        else:
            self._mixed_obs = False
            self.action_base = MLPBase(
                cfg,
                action_shape,
                use_attn_internal=cfg.use_attn_internal,
                use_cat_self=True,
            )

        # state_input_size = self.state_base.output_size
        # action_input_size = self.action_base.output_size
        # input_size = state_input_size + action_input_size
        input_size = self.state_base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                rnn_type=cfg.rnn_type,
            )
            # input_size = self.hidden_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.combine_feature_layer_1 = init_(nn.Linear(input_size, self.hidden_size))
        self.combine_feature_layer_2 = init_(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.combine_feature_layer_3 = init_(nn.Linear(self.hidden_size, 1))

        if use_half:
            self.half()
        self.to(device)

    def forward(self, state, action, rnn_states, masks):
        if self._mixed_obs:
            for key in state.keys():
                state[key] = check(state[key]).to(**self.tpdv)
            for key in action.keys():
                action[key] = check(action[key]).to(**self.tpdv)
        else:
            state = check(state).to(**self.tpdv)
            action = check(action).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        state_feature = F.relu(self.state_base(state))
        action_feature = F.relu(self.action_base(action))
        features = state_feature + action_feature

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        critic_out = self.combine_feature_layer_1(features)
        critic_out = F.relu(self.combine_feature_layer_2(critic_out))
        critic_out = self.combine_feature_layer_3(critic_out)

        return critic_out, rnn_states
