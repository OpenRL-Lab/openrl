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

import pytest
import torch

from openrl.datasets.expert_dataset import ExpertDataset
from openrl.envs.common import make
from openrl.envs.vec_env.wrappers.gen_data import GenDataWrapper
from openrl.envs.wrappers.monitor import Monitor

env_wrappers = [
    Monitor,
]


def gen_data(total_episode, data_save_path):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.

    env = make(
        "IdentityEnv",
        env_num=1,
        asynchronous=True,
        env_wrappers=env_wrappers,
    )

    env = GenDataWrapper(
        env, data_save_path=data_save_path, total_episode=total_episode
    )
    env.reset()
    done = False
    ep_length = 0
    while not done:
        obs, r, done, info = env.step(env.random_action())
        ep_length += 1
    env.close()
    return ep_length


@pytest.mark.unittest
def test_expert_dataset(tmp_path):
    total_episode = 1
    data_save_path = tmp_path / "data.pkl"
    ep_length = gen_data(total_episode, data_save_path)

    dataset = ExpertDataset(
        data_save_path,
        num_trajectories=None,
        subsample_frequency=1,
        seed=None,
        env_id=0,
        env_num=1,
    )
    assert len(dataset) == ep_length, "len(dataset)={},data_length={}".format(
        len(dataset), ep_length
    )
    assert len(dataset[0]) == 2, "len(dataset[0])={}".format(len(dataset[0]))

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, drop_last=True
    )

    step = 0
    for batch_data in data_loader:
        assert len(batch_data) == 2, "len(batch_data)={}".format(len(batch_data))
        step += 1
    assert step == ep_length, "step={},ep_length={}".format(step, ep_length)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
