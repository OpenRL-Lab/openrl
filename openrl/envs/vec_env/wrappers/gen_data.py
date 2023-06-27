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
from gymnasium.core import ActType
from tqdm.rich import tqdm

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.wrappers.base_wrapper import VecEnvWrapper
from openrl.envs.wrappers.monitor import Monitor


class GenDataWrapper(VecEnvWrapper):
    def __init__(self, env: BaseVecEnv, data_save_path: str, total_episode: int):
        assert env.env_is_wrapped(Monitor, indices=0)[0]

        super().__init__(env)

        self.data_save_path = data_save_path
        self.total_episode = total_episode
        self.pbar = None

    def reset(self, **kwargs):
        self.current_total_episode = 0
        self.episode_lengths = []

        self.data = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "info": [],
        }

        returns = self.env.reset(**kwargs)
        if len(returns) == 2:
            obs = returns[0]
            info = returns[1]
        else:
            obs = returns
            info = {}
        self.data["action"].append(None)
        self.data["obs"].append(obs)
        self.data["reward"].append(None)
        self.data["done"].append(None)
        self.data["info"].append(info)
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()
        self.pbar = tqdm(total=self.total_episode)

        return returns

    def step(self, action: ActType, *args, **kwargs):
        self.data["action"].append(action)
        obs, r, done, info = self.env.step(action, *args, **kwargs)
        self.data["obs"].append(obs)
        self.data["reward"].append(r)
        self.data["done"].append(done)
        self.data["info"].append(info)
        for i in range(self.env.parallel_env_num):
            if np.all(done[i]):
                self.current_total_episode += 1
                self.pbar.update(1)
                assert "final_info" in info[i] and "episode" in info[i]["final_info"]
                self.episode_lengths.append(info[i]["final_info"]["episode"]["l"])

        done = self.current_total_episode >= self.total_episode

        return obs, r, done, info

    def close(self, **kwargs):
        self.pbar.refresh()
        self.pbar.close()
        average_length = np.mean(self.episode_lengths)
        print("collect total episode: {}".format(self.current_total_episode))
        print("average episode length: {}".format(average_length))

        self.data["total_episode"] = self.current_total_episode
        self.data["average_length"] = average_length
        pickle.dump(self.data, open("data.pkl", "wb"))
        return self.env.close(**kwargs)
