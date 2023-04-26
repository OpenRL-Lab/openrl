#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
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
from io import StringIO
from pathlib import Path
from typing import Optional

from datasets import load_from_disk


def data_abs_path(path: str, data_server_dir: Optional[str] = None) -> str:
    if "data_server://" in path:
        if data_server_dir is None:
            data_server_dir = Path.home() / "data_server/"
        if type(data_server_dir) == str:
            data_server_dir = Path(data_server_dir)
        assert (
            data_server_dir.is_dir()
        ), "Can not find data_server directory at: {}".format(data_server_dir)
        return path.replace("data_server:/", str(data_server_dir))
    else:
        return path


def replace_data_server(text: str, data_server_dir: Optional[str] = None) -> str:
    if data_server_dir is None:
        data_server_dir = str(Path.home() / "data_server/")

    return text.replace("data_server:/", data_server_dir)


def data_server_wrapper(fp):
    text = replace_data_server(fp.read())
    return StringIO(text)


def load_dataset(data_path: str, split: str):
    if Path(data_path).exists():
        dataset = load_from_disk("{}/{}".format(data_path, split))
    elif "data_server:" in data_path:
        data_path = data_path.split("data_server:")[-1]
        dataset = load_from_disk(
            Path.home()
            / "data_server/huggingface/datasets/{}/{}".format(data_path, split)
        )
    else:
        from datasets import load_dataset

        dataset = load_dataset(data_path, split=split)
    return dataset
