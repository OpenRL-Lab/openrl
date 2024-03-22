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
from pathlib import Path

import pytest

from openrl.supports.opendata.utils.opendata_utils import data_abs_path, load_dataset


@pytest.mark.unittest
def test_data_abs_path(tmpdir):
    data_path = "./"
    assert data_abs_path(data_path) == data_path

    data_server_dir = Path.home() / "data_server/"

    new_create = False
    if not data_server_dir.exists():
        data_server_dir.mkdir()
        new_create = True
    data_abs_path("data_server://data_path")
    if new_create:
        data_server_dir.rmdir()
    data_abs_path("data_server://data_path", str(tmpdir))


@pytest.mark.unittest
def test_load_dataset(tmpdir):
    try:
        load_dataset(str(tmpdir), "train")
    except Exception as e:
        pass
    try:
        load_dataset("data_server://data_path", "train")
    except Exception as e:
        pass
    try:
        load_dataset(str(tmpdir) + "/test", "train")
    except Exception as e:
        pass


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
