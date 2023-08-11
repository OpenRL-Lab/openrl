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

from run_arena import run_arena


def test_seed(seed: int):
    test_time = 5
    pre_result = None
    for parallel in [False, True]:
        for i in range(test_time):
            result = run_arena(seed=seed, parallel=parallel, total_games=20)
            if pre_result is not None:
                assert pre_result == result, f"parallel={parallel}, seed={seed}"
            pre_result = result


if __name__ == "__main__":
    test_seed(0)
