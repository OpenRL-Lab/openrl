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
import copy
import pickle

import numpy as np
import torch.utils.data


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(
        self, file_name, num_trajectories=None, subsample_frequency=1, seed=None
    ):
        # if num_trajectories=4, subsample_frequency=20, then use the data of 4 trajectories, and the size of the data of each trajectory is reduced by 20 times
        if seed is not None:
            torch.manual_seed(seed)
        all_trajectories = pickle.load(open(file_name, "rb"))
        if num_trajectories is None:
            num_trajectories = len(all_trajectories["episode_lengths"])

        perm = torch.randperm(len(all_trajectories["episode_lengths"]))

        idx = perm[:num_trajectories]

        self.trajectories = {}

        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories,)
        ).long()

        for k, v in all_trajectories.items():
            data = [v[ii] for ii in idx]

            if k != "episode_lengths" and k != "episode_rewards":
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][start_idx[i] :: subsample_frequency])
                self.trajectories[k] = samples

            elif k == "episode_lengths":
                self.trajectories[k] = np.array(
                    [data[i] // subsample_frequency for i in range(num_trajectories)]
                )

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = np.sum(self.trajectories["episode_lengths"])
        print("total data length:", self.length)

        traj_idx = 0
        i = 0

        self.get_idx = []

        for j in range(self.length):
            while self.trajectories["episode_lengths"][traj_idx].item() <= i:
                i -= self.trajectories["episode_lengths"][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, step_i = self.get_idx[i]
        # print(i,"traj_idx", traj_idx,step_i, self.trajectories["obs"][traj_idx][step_i])
        return (
            self.trajectories["obs"][traj_idx][step_i],
            self.trajectories["action"][traj_idx][step_i],
        )
