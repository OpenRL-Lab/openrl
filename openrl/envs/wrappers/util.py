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

from typing import Any

import numpy as np
from gymnasium.spaces.box import Box


def nest_expand_dim(input: Any) -> Any:
    if isinstance(input, (np.ndarray, float, int)):
        return np.expand_dims(input, 0)
    elif isinstance(input, list):
        return [input]
    elif isinstance(input, dict):
        for key in input:
            input[key] = nest_expand_dim(input[key])
        return input
    elif isinstance(input, Box):
        return [input]
    else:
        raise NotImplementedError("Not support type: {}".format(type(input)))
