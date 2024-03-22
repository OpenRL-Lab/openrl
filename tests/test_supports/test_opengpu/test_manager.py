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

import argparse
import os
import sys

import pytest

from openrl.supports.opengpu.manager import LocalGPUManager


@pytest.fixture(
    scope="module",
    params=[
        # 添加不同的参数组合以进行测试
        0,
        1,
        2,
        None,
    ],
)
def learner_num(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def disable_cuda(request):
    return request.param


@pytest.fixture(scope="module", params=["all", "single", "error_type"])
def gpu_usage_type(request):
    return request.param


@pytest.fixture(
    scope="module",
)
def args(learner_num, disable_cuda, gpu_usage_type):
    if learner_num is None:
        return None
    current_dict = {}
    current_dict["learner_num"] = learner_num
    current_dict["disable_cuda"] = disable_cuda
    current_dict["gpu_usage_type"] = gpu_usage_type

    return argparse.Namespace(**current_dict)


@pytest.mark.unittest
def test_local_manager(args):
    manager = LocalGPUManager(args)
    manager.get_gpu()
    try:
        manager.get_learner_gpu()
    except IndexError as e:
        print("Caught an IndexError:", e)
    try:
        assert isinstance(manager.get_learner_gpus(), list)
    except IndexError as e:
        print("Caught an IndexError:", e)
    manager.get_worker_gpu()
    manager.log_info()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
