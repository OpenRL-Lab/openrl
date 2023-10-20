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

import os
import sys

import numpy as np
import pytest
from gymnasium import spaces

from openrl.configs.config import create_config_parser
from openrl.modules.networks.policy_value_network_gpt import (
    PolicyValueNetworkGPT as PolicyValueNetwork,
)


@pytest.fixture(scope="module", params=["--model_path test_gpt2"])
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_gpt_network(config):
    net = PolicyValueNetwork(
        cfg=config,
        input_space=spaces.Discrete(2),
        action_space=spaces.Discrete(2),
    )

    net.get_actor_para()
    net.get_critic_para()

    obs = {
        "input_encoded_pt": np.zeros([1, 2]),
        "input_attention_mask_pt": np.zeros([1, 2]),
    }
    rnn_states = np.zeros(2)
    masks = np.zeros(2)
    action = np.zeros(1)
    net.get_actions(obs=obs, rnn_states=rnn_states, masks=masks)
    net.eval_actions(
        obs=obs, rnn_states=rnn_states, action=action, masks=masks, action_masks=None
    )
    net.get_values(obs=obs, rnn_states=rnn_states, masks=masks)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
