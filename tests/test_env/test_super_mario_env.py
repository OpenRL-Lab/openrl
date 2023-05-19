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


@pytest.mark.unittest
def test_super_mario():
    from openrl.envs.common import make

    env_num = 2
    env = make("SuperMarioBros-1-1-v1", env_num=env_num)
    obs, info = env.reset()
    obs, reward, done, info = env.step(env.random_action())

    assert obs["critic"].shape[2] == 3
    assert env.parallel_env_num == env_num

    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
