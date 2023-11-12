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
from typing import Any, Optional, Dict

import numpy as np
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
from openrl.envs.nlp.utils.distribution import CategoricalDistribution

from transformers.modeling_utils import unwrap_model

class PolicyNetworkGPT(BasePolicyNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        disable_drop_out: bool = True,
        extra_args=None,
    ) -> None:
        
        self.use_half = use_half
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        super(PolicyNetworkGPT, self).__init__(cfg, device)
        
        self.disable_drop_out = disable_drop_out
        
        
        
        self._action_dist = CategoricalDistribution(action_space.n)
        
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(cfg.model_path)
        config_dict = config.to_dict()
        for key in config_dict:
            if "drop" in key:
                config_dict[key] = 0.0
        config = config.from_dict(config_dict)
        self._policy_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, config=config
        )
        self._policy_model.config.use_cache = False

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.forward_original(*args, **kwargs)
        elif forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        else:
            raise NotImplementedError
        
    def _prepare_inputs_for_model(
        self,
        model: Any,
        input_ids: torch.tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )
        return model_inputs

    def forward_original(
        self, raw_obs, rnn_states, masks, action_masks=None, deterministic=False
    ):
        for key in raw_obs.keys():
            raw_obs[key] = torch.from_numpy(raw_obs[key]) if type(raw_obs[key]) == np.ndarray else raw_obs[key]
            raw_obs[key] = raw_obs[key].to(self._policy_model.device)
            # raw_obs[key] = check(raw_obs[key], self.use_half, self.tpdv)
            # if self._use_fp16:
            #     raw_obs[key] = raw_obs[key].half()
        rnn_states = check(rnn_states)
        
        input_ids = raw_obs["input_encoded_pt"].int()
        attention_mask = raw_obs["input_attention_mask_pt"]
        
        past_model_kwargs = None
        
        if past_model_kwargs is None:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        
        model_inputs = self._prepare_inputs_for_model(
            self._policy_model, input_ids, past_model_kwargs
        )
        
        # forward pass to transformers
        output = self._policy_model(**model_inputs)
        
        # compute action probs - policy head
        next_token_logits = output.logits[:, -1]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        
        actions = dist.mode() if deterministic else dist.sample()
        action_log_probs = dist.log_prob(actions)
        
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1), rnn_states

    def eval_actions(
        self, obs, rnn_states, action, masks, action_masks=None, active_masks=None
    ):
        for key in obs.keys():
            obs[key] = torch.from_numpy(obs[key]) if type(obs[key]) == np.ndarray else obs[key]
            obs[key] = obs[key].to(self._policy_model.device)
            # obs[key] = check(obs[key], self.use_half, self.tpdv)
            # if self._use_fp16:
            #     obs[key] = obs[key].half()
        action = check(action).to(self._policy_model.device).squeeze()
        rnn_states = check(rnn_states)
        
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]
        
        past_model_kwargs = None
        
        if past_model_kwargs is None:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        
        model_inputs = self._prepare_inputs_for_model(
            self._policy_model, input_ids, past_model_kwargs
        )
        
        # forward pass to transformers
        output = self._policy_model(**model_inputs)
        
        # compute action probs - policy head
        next_token_logits = output.logits[:, -1]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = None

        return action_log_probs.unsqueeze(-1), dist_entropy.mean(), values

    def get_policy_values(self, obs, rnn_states, masks):
        raise NotImplementedError