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
from typing import List, Optional, Union

import envpool

from openrl.envs.common import build_envs


def make_envpool_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    assert "env_type" in kwargs
    assert kwargs.get("env_type") in ["gym", "dm", "gymnasium"]
    # Since render_mode is not supported, we set envpool to True
    # so that we can remove render_mode keyword argument from build_envs
    assert render_mode is None, "envpool does not support render_mode yet"
    kwargs["envpool"] = True

    env_wrappers = kwargs.pop("env_wrappers")
    env_fns = build_envs(
        make=envpool.make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
