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
from openrl.modules.networks.MAT_network import MultiAgentTransformer


@pytest.fixture(scope="module", params=[""])
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_MAT_network(config):
    net = MultiAgentTransformer(
        config,
        input_space=spaces.Discrete(2),
        action_space=spaces.Discrete(2),
    )
    net.get_actor_para()
    net.get_critic_para()

    obs = np.zeros([1, 2])
    rnn_states = np.zeros(2)
    masks = np.zeros(2)
    action = np.zeros(1)
    net.get_actions(obs=obs, masks=masks)
    net.eval_actions(
        obs=obs, rnn_states=rnn_states, action=action, masks=masks, action_masks=None
    )
    net.get_values(critic_obs=obs, masks=masks)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
