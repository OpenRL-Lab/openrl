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
import pickle
import sys

import numpy as np
import pytest

from openrl.envs.common import make
from openrl.envs.vec_env.wrappers.gen_data import GenDataWrapper, GenDataWrapper_v1
from openrl.envs.vec_env.wrappers.zero_reward_wrapper import ZeroRewardWrapper
from openrl.envs.wrappers.monitor import Monitor


@pytest.mark.unittest
def test_zero_reward_wrapper():
    env = make("IdentityEnv", env_num=1)
    env = ZeroRewardWrapper(env)
    env.reset(seed=0)
    while True:
        obs, reward, done, info = env.step(env.random_action())
        assert np.all(reward == 0), "reward should be zero"
        if done:
            break
    env.close()


@pytest.mark.unittest
def test_gen_data(tmp_path):
    total_episode = 4
    env = make("IdentityEnv", env_wrappers=[Monitor], env_num=1)
    data_save_path = tmp_path / "data.pkl"
    env = GenDataWrapper(
        env, data_save_path=str(data_save_path), total_episode=total_episode
    )
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, r, done, info = env.step(env.random_action())
    env.close()

    save_data = pickle.load(open(data_save_path, "rb"))
    assert len(save_data["episode_lengths"]) == total_episode, (
        f"episode_lengths {len(save_data['episode_lengths'])} "
        f"should be equal to total_episode {total_episode}"
    )


@pytest.mark.unittest
def test_gen_data_old(tmp_path):
    total_episode = 4
    env = make("IdentityEnv", env_wrappers=[Monitor], env_num=1)
    data_save_path = tmp_path / "data.pkl"
    env = GenDataWrapper_v1(
        env, data_save_path=str(data_save_path), total_episode=total_episode
    )
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, r, done, info = env.step(env.random_action())
    env.close()

    save_data = pickle.load(open(data_save_path, "rb"))
    assert save_data["total_episode"] == total_episode, (
        f"episode_lengths {save_data['total_episode']} "
        f"should be equal to total_episode {total_episode}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
