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
from gymnasium.wrappers import EnvCompatibility

from openrl.envs.toy_envs import make_toy_envs
from openrl.envs.vec_env.sync_venv import SyncVectorEnv


class CustomEnvCompatibility(EnvCompatibility):
    def reset(self, **kwargs):
        return super().reset(**kwargs)[0]


def init_envs():
    env_wrappers = [CustomEnvCompatibility]
    env_fns = make_toy_envs(
        id="IdentityEnv",
        env_num=2,
        env_wrappers=env_wrappers,
    )
    return env_fns


def assert_env_name(env, env_name):
    assert env.metadata["name"].__name__ == env_name


@pytest.mark.unittest
def test_sync_env():
    env_name = "IdentityEnv"
    env = SyncVectorEnv(init_envs())
    env.exec_func(assert_env_name, indices=None, env_name=env_name)
    env.call("render")


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
