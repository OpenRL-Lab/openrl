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

source_dir = Path("./api_docs")
assert source_dir.exists(), f"{source_dir} not exists"
def change_openrl_tile():
    # 修改openrl.rst 的第一行title
    openrl_rst = source_dir / "openrl.rst"
    with open(openrl_rst, "r") as f:
        lines = f.readlines()
    lines[0] = "API Doc\n"

    for i in range(len(lines)):
        if "Module contents" in lines[i]:
            del lines[i:]
            break

    with open(openrl_rst, "w") as f:
        f.writelines(lines)

def remove_files(file_list):
    remmoved_files = []
    unremovete_files = []
    for file in file_list:
        file_path = source_dir / file
        if file_path.exists():
            file_path.unlink()
            remmoved_files.append(file)
        else:
            unremovete_files.append(file)
    if len(remmoved_files) > 0: print("Removed files: ", remmoved_files)
    if len(unremovete_files)> 0: print("Un-removed files: ", unremovete_files)

if __name__ == '__main__':
    change_openrl_tile()
    remove_files(["modules.rst"])