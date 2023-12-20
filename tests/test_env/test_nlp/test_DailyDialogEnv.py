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

from openrl.configs.config import create_config_parser
from openrl.envs.common import make


@pytest.fixture(
    scope="module",
    params=["--env.args {'data_path':None,'tokenizer_path':'builtin_BPE'}"],
)
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_DailyDialogEnv(config):
    env = make("daily_dialog", env_num=1, asynchronous=False, cfg=config)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
