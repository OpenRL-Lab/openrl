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

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from openrl.selfplay.opponents.base_opponent import BaseOpponent


def check_opponent_template(opponent_template: Union[str, Path]):
    if isinstance(opponent_template, str):
        opponent_template = Path(opponent_template)
    assert (
        opponent_template.exists()
    ), f"opponent_template {opponent_template} does not exist"
    info_file = opponent_template / "info.json"
    assert (
        info_file.exists()
    ), f"opponent_template {opponent_template} does not contain info.json"
    opponent_script = opponent_template / "opponent.py"
    assert (
        opponent_script.exists()
    ), f"opponent_template {opponent_template} does not contain opponent.py"


def load_opponent_from_path(
    opponent_path: Union[str, Path], opponent_info: Optional[Dict[str, str]] = None
) -> Optional[BaseOpponent]:
    opponent = None
    if isinstance(opponent_path, str):
        opponent_path = Path(opponent_path)
    try:
        sys.path.append(str(opponent_path.parent))
        opponent_module = __import__(
            "{}.opponent".format(opponent_path.name), fromlist=["opponent"]
        )
        opponent = opponent_module.Opponent(
            opponent_id=opponent_info["opponent_id"],
            opponent_path=opponent_path,
            opponent_info=opponent_info,
        )
    except:
        print(f"Load opponent from {opponent_path} failed")

    sys.path.remove(str(opponent_path.parent))
    return opponent


def get_opponent_from_path(
    opponent_path: Union[str, Path],
    current_opponent: Optional[BaseOpponent] = None,
    lazy_load_opponent: bool = True,
    opponent_info: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[BaseOpponent], bool]:
    if isinstance(opponent_path, str):
        opponent_path = Path(opponent_path)
    assert opponent_path.exists(), f"opponent_path {opponent_path} does not exist"
    info_file = opponent_path / "info.json"
    assert (
        info_file.exists()
    ), f"opponent_path {opponent_path} does not contain info.json"
    opponent_script = opponent_path / "opponent.py"
    assert (
        opponent_script.exists()
    ), f"opponent_path {opponent_path} does not contain opponent.py"

    if opponent_info is None:
        with open(info_file) as f:
            opponent_info = json.load(f)

    opponent_type = opponent_info["opponent_type"]

    is_new_opponent = False
    if (
        current_opponent is not None
        and current_opponent.opponent_type == opponent_type
        and lazy_load_opponent
    ):
        return current_opponent.load(opponent_path), is_new_opponent
    else:
        is_new_opponent = True
        return load_opponent_from_path(opponent_path, opponent_info), is_new_opponent


def get_opponent_from_info(
    opponent_info: Dict[str, str],
    current_opponent: Optional[BaseOpponent] = None,
    lazy_load_opponent: bool = True,
) -> Tuple[Optional[BaseOpponent], bool]:
    opponent_id = opponent_info["opponent_id"]
    opponent_path = opponent_info["opponent_path"]
    opponent_type = opponent_info["opponent_type"]
    is_new_opponent = False
    if current_opponent is not None and current_opponent.opponent_id == opponent_id:
        return current_opponent, is_new_opponent

    return get_opponent_from_path(
        opponent_path, current_opponent, lazy_load_opponent, opponent_info
    )
