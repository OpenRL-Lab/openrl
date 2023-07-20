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

import pickle

import numpy as np
import torch.utils.data


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name,
        num_trajectories=None,
        subsample_frequency=1,
        seed=None,
        env_id=0,
        env_num=1,
    ):
        # if num_trajectories=4, subsample_frequency=20, then use the data of 4 trajectories, and the size of the data of each trajectory is reduced by 20 times
        if seed is not None:
            torch.manual_seed(seed)
        assert num_trajectories is None or env_num == 1
        assert (
            env_id < env_num
        ), "env_id must be less than env_num, but got env_id={}, env_num={}".format(
            env_id, env_num
        )
        self.env_id = env_id
        self.env_num = env_num

        all_trajectories = pickle.load(open(file_name, "rb"))

        if num_trajectories is None:
            all_trajectory_num = len(all_trajectories["episode_lengths"])
            assert env_num <= all_trajectory_num, (
                "env_num must be less than all_trajectory_num, but got env_num={},"
                " all_trajectory_num={}".format(env_num, all_trajectory_num)
            )
            start_traj_idx = all_trajectory_num // env_num * env_id
            end_traj_idx = all_trajectory_num // env_num * (env_id + 1)
        else:
            start_traj_idx = 0
            end_traj_idx = num_trajectories

        num_trajectories = end_traj_idx - start_traj_idx

        perm = torch.randperm(len(all_trajectories["episode_lengths"]))

        if "env_info" in all_trajectories:
            if "observation_space" in all_trajectories["env_info"]:
                self.observation_space = all_trajectories["env_info"][
                    "observation_space"
                ]
            else:
                self.observation_space = None  # get observation space from obs data
            if "action_space" in all_trajectories["env_info"]:
                self.action_space = all_trajectories["env_info"]["action_space"]
            else:
                self.action_space = None  # get action space from action data
            if "agent_num" in all_trajectories["env_info"]:
                self.agent_num = all_trajectories["env_info"]["agent_num"]
            else:
                self.agent_num = None  # get agent num from obs data
            del all_trajectories["env_info"]

        idx = perm[start_traj_idx:end_traj_idx]

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
        return (
            self.trajectories["obs"][traj_idx][step_i],
            self.trajectories["action"][traj_idx][step_i],
        )
