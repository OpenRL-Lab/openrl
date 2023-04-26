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

import numpy as np
import pytest

from openrl.buffers.utils.obs_data import ObsData


@pytest.mark.unittest
def test_basic():
    a_data = np.array([1, 2])
    b_data = np.array([3, 4])
    c_data = np.array([5, 6, 7])
    obs = ObsData({"a": a_data, "b": b_data, "c": c_data})

    assert np.all(obs["c"] == c_data)
    obs["a"][0] = 99
    a_data[0] = 99
    assert np.all(obs["a"] == a_data)

    assert np.all(obs.flatten() == np.concatenate([a_data, b_data, c_data]))


@pytest.mark.unittest
def test_step():
    obs_stepes = [
        {"obs_a": np.array([[[0, 1]]]), "obs_b": np.array([[[3, 9]]])},
        {"obs_a": np.array([[[2, 4]]]), "obs_b": np.array([[[6, 8]]])},
    ]
    obs = ObsData({"obs_a": np.zeros((2, 1, 1, 2)), "obs_b": np.zeros((2, 1, 1, 2))})
    for step in range(len(obs_stepes)):
        for key in obs.keys():
            obs[key][step] = obs_stepes[step][key]

    step_data = {"obs_a": np.array([[0.0, 1.0]]), "obs_b": np.array([[3.0, 9.0]])}
    for key in obs[0]:
        assert np.all(obs[0][key] == step_data[key])


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
