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
from pathlib import Path
from typing import Any, Dict, List, Union

from openrl.selfplay.opponents.utils import check_opponent_template
from openrl.utils.file_tool import copy_files, link_files


class OpponentTemplate:
    def __init__(
        self, opponent_template: Union[str, Path], copy_script_file: bool = False
    ):
        if isinstance(opponent_template, str):
            opponent_template = Path(opponent_template)
        self.opponent_template = opponent_template
        self.copy_script_file = copy_script_file
        self.check()
        self.opponent_info = self.load_opponent_info()
        self.script_files = self.load_script_files()

    def load_opponent_info(self) -> Dict[str, Any]:
        info_file = self.opponent_template / "info.json"
        assert (
            info_file.exists()
        ), f"opponent_template {self.opponent_template} does not contain info.json"
        return json.loads(info_file.read_text())

    def load_script_files(self) -> Union[List[str], List[Path]]:
        script_files = list(self.opponent_template.glob("**/*.py"))
        return [f.absolute() for f in script_files]

    def check(self):
        check_opponent_template(self.opponent_template)

    def save(self, opponent_path: Union[str, Path], opponent_info: Dict[str, Any]):
        if isinstance(opponent_path, str):
            opponent_path = Path(opponent_path)
        assert opponent_path.exists(), f"opponent_path {opponent_path} does not exist"

        if self.copy_script_file:
            copy_files(self.script_files, opponent_path)
        else:
            link_files(self.script_files, opponent_path)

        new_opponent_info = self.opponent_info.copy()
        new_opponent_info.update(opponent_info)
        with (opponent_path / "info.json").open("w", encoding="utf-8") as f:
            json.dump(new_opponent_info, f, ensure_ascii=False, indent=4)
