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
import torch

from openrl.datasets.expert_dataset import ExpertDataset


def test_dataset():
    dataset = ExpertDataset(file_name="data_small.pkl", seed=0)
    print("data length:", len(dataset))
    print("data[0]:", dataset[0][0])
    print("data[1]:", dataset[1][0])
    print("data[len(data)-1]:", dataset[len(dataset) - 1][0])

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=128, shuffle=False, drop_last=True
    )
    for batch_data in data_loader:
        expert_obs, expert_action = batch_data


if __name__ == "__main__":
    test_dataset()
