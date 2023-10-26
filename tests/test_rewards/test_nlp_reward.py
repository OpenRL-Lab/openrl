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

from openrl.buffers.normal_buffer import NormalReplayBuffer
from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.rewards import RewardFactory


@pytest.fixture(
    scope="module",
    params=[
        "--reward_class.id  NLPReward --reward_class.args"
        " {'intent_model':'builtin_intent','ref_model':'builtin_ref','use_deepspeed':False}"
    ],
)
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_nlp_reward(config):
    env = make("fake_dialog_data", env_num=1)
    reward = RewardFactory.get_reward_class(config.reward_class, env)
    data = {}
    data["rewards"] = np.zeros(32)
    env_info = {}
    env_info["final_info"] = {
        "prompt_texts": "hello",
        "generated_texts": "hello",
        "meta_infos": {"intent": [1]},
    }
    data["infos"] = [env_info] * 32
    data["step"] = 0
    data["actions"] = [0]
    data["action_log_probs"] = np.zeros(32)
    buffer = NormalReplayBuffer(
        config,
        num_agents=env.agent_num,
        obs_space=env.observation_space,
        act_space=env.action_space,
        data_client=None,
        episode_length=1,
    )
    data["buffer"] = buffer
    reward.step_reward(data=data)
    reward.batch_rewards(buffer=buffer)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
