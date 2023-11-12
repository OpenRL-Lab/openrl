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

from openrl.buffers.utils.util import get_critic_obs_space
from openrl.modules.networks.base_value_network import BaseValueNetwork
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase, MLPLayer
from openrl.modules.networks.utils.popart import PopArt
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.modules.utils.valuenorm import ValueNorm
from openrl.utils.util import check_v2 as check

from transformers.modeling_utils import unwrap_model

class ValueNetworkGPT(BaseValueNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space=None,
        use_half=False,
        device=torch.device("cpu"),
        extra_args=None,
    ):
        
        self.use_half = use_half
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        super(ValueNetworkGPT, self).__init__(cfg, device)
        
        from transformers import AutoModelForCausalLM
        
        self._value_model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
        self._value_model.config.use_cache = False
        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False
        )
        self.value_normalizer = (
            ValueNorm(1, device=device) if self._use_valuenorm else None
        )
        
        self._value_head.to(self.device)

        
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

    def forward(self, critic_obs, rnn_states, masks):
        for key in critic_obs.keys():
            critic_obs[key] = torch.from_numpy(critic_obs[key]) if type(critic_obs[key]) == np.ndarray else critic_obs[key]
            critic_obs[key] = critic_obs[key].to(self._value_model.device)
            # critic_obs[key] = check(critic_obs[key], self.use_half, self.tpdv)
            # if self._use_fp16:
            #     critic_obs[key] = critic_obs[key].half()
        masks = check(masks).to(self._value_model.device)
        rnn_states = check(rnn_states)
        
        input_ids = critic_obs["input_encoded_pt"].int()
        attention_mask = critic_obs["input_attention_mask_pt"]
        
        past_model_kwargs = None
        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        
        model_inputs = self._prepare_inputs_for_model(
            self._value_model, input_ids, past_model_kwargs
        )
        output = self._value_model(output_hidden_states=True, **model_inputs)
        last_tokens_hidden = output.hidden_states[-1][:, -1]
        values = self._value_head.forward(last_tokens_hidden)

        return values, rnn_states
