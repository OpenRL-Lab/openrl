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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openrl.buffers.utils.util import get_critic_obs_space, get_policy_obs_space
from openrl.modules.networks.base_policy_network import BasePolicyNetwork
from openrl.modules.networks.base_value_network import BaseValueNetwork
from openrl.modules.networks.ddpg_network import ActorNetwork
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class SACActorNetwork(ActorNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        extra_args=None,
        log_std_min=-20,
        log_std_max=2,
    ) -> None:
        super().__init__(cfg, input_space, action_space, device, use_half, extra_args)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]
        input_size = self.base.output_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if isinstance(self.action_space, gym.spaces.box.Box):
            self.actor_out = init_(nn.Linear(input_size, action_space.shape[0] * 2))
        else:
            raise NotImplementedError(
                f"This type ({type(self.action_space)}) of game has not been"
                " implemented."
            )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        features = self.base(obs)

        if isinstance(self.action_space, gym.spaces.box.Box):
            output = self.actor_out(features)
            # print(output)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        else:
            raise NotImplementedError("This type of game has not been implemented.")

        return mean, log_std

    def _normalize(self, action) -> torch.Tensor:
        """
        Normalize the action value to the action space range.
        the return values of self.fcs is between -1 and 1 since we use tanh as output activation, while we want the action ranges to be (self.action_space.low, self.action_space.high).
        """
        # print(self.action_space.high, self.action_space.low)
        # exit()

        return action
        # return torch.clamp(
        #     action,
        #     torch.tensor(self.action_space.low).detach(),
        #     torch.tensor(self.action_space.high).detach(),
        # )
        # action = (action + 1) / 2 * (
        #     torch.tensor(self.action_space.high) - torch.tensor(self.action_space.low)
        # ) + torch.tensor(self.action_space.low)
        # return action

    def evaluate(self, obs, deterministic=True):
        mean, log_std = self.forward(obs)

        if deterministic:
            # action = torch.tanh(mean)  # add tanh to activate
            action = mean
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(axis=-1)
            log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
                axis=-1
            )
            return self._normalize(action), log_prob.unsqueeze(dim=-1)

        # sample action from N(mean, std) if sample is True
        # obtain log_prob for policy and Q function update
        # use the reparameterization trick, and perform tanh normalization

        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            axis=-1
        )  # NOTE: The correction formula from the original SAC paper (arXiv 1801.01290) appendix C
        # action = torch.tanh(action)  # add tanh to activate

        return self._normalize(action), log_prob.unsqueeze(dim=-1)
