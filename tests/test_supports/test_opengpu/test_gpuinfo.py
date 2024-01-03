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

from openrl.supports.opengpu.gpu_info import preserve_decimal


@pytest.mark.unittest
def test_preserve_decimal():
    preserve_decimal(1, 2)
    preserve_decimal(1.1, 0)
    preserve_decimal(1.1, -1)
    preserve_decimal(1.1, 4)
    preserve_decimal(-1.1, 4)
    preserve_decimal(-0.1, 0)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
