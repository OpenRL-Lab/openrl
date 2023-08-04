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

from pathlib import Path
from typing import List, Union


def copy_files(
    source_files: Union[List[str], List[Path]], target_dir: Union[str, Path]
):
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        if isinstance(source_file, str):
            source_file = Path(source_file)
        target_file = target_dir / source_file.name
        target_file.write_text(source_file.read_text())


def link_files(
    source_files: Union[List[str], List[Path]], target_dir: Union[str, Path]
):
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        if isinstance(source_file, str):
            source_file = Path(source_file)
        target_file = target_dir / source_file.name
        target_file.symlink_to(source_file)
