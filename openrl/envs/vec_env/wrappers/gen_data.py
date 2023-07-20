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
from gymnasium.core import ActType
from tqdm.rich import tqdm

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.wrappers.base_wrapper import VecEnvWrapper
from openrl.envs.wrappers.monitor import Monitor


class TrajectoryData:
    def __init__(
        self, env_num, total_episode, observation_space, action_space, agent_num: int
    ):
        self.env_num = env_num
        self.all_keys = ["obs", "action", "reward", "done", "info"]
        self.total_episode = total_episode
        self.all_trajectories = None
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_num = agent_num

    def init_empty_dict(self, source_dict={}):
        for key in self.all_keys:
            source_dict[key] = []
        return copy.copy(source_dict)

    def reset(self, reset_data):
        self.current_total_episode = 0
        self.episode_lengths = []
        self.episode_rewards = []

        self.all_trajectories = self.init_empty_dict()

        self.data = [self.init_empty_dict() for _ in range(self.env_num)]
        for key in reset_data:
            for i in range(self.env_num):
                self.data[i][key].append(copy.copy(reset_data[key][i]))

    def step(self, step_data):
        step_data = copy.copy(step_data)
        finished = []
        need_finish = False
        for i in range(self.env_num):
            if need_finish:
                break
            done = step_data["done"][i]
            if np.all(done):
                assert (
                    "final_info" in step_data["info"][i]
                    and "episode" in step_data["info"][i]["final_info"]
                )
                self.episode_lengths.append(
                    step_data["info"][i]["final_info"]["episode"]["l"]
                )
                self.episode_rewards.append(
                    step_data["info"][i]["final_info"]["episode"]["r"]
                )

                finished.append(i)

                # step_data["obs"][i] = step_data["info"][i]["final_observation"]
                # step_data["info"][i] = step_data["info"][i]["final_info"]

                for key in self.data[i]:
                    if key in ["obs", "info"]:
                        if key == "obs":
                            self.data[i][key].append(
                                copy.copy(step_data["info"][i]["final_observation"])
                            )
                        elif key == "info":
                            self.data[i][key].append(
                                copy.copy(step_data["info"][i]["final_info"])
                            )
                        assert len(self.data[i][key]) == self.episode_lengths[-1] + 1, (
                            f"key: {key}, len: {len(self.data[i][key])},"
                            f" episode_lengths: {self.episode_lengths[-1]}"
                        )
                    elif key in ["action", "reward", "done"]:
                        self.data[i][key].append(copy.copy(step_data[key][i]))
                        assert len(self.data[i][key]) == self.episode_lengths[-1], (
                            f"key: {key}, len: {len(self.data[i][key])},"
                            f" episode_lengths: {self.episode_lengths[-1]}"
                        )
                    self.all_trajectories[key].append(copy.copy(self.data[i][key]))

                self.data[i] = self.init_empty_dict()

                del step_data["info"][i]["final_observation"]
                del step_data["info"][i]["final_info"]

                for key in ["obs", "info"]:
                    self.data[i][key].append(copy.copy(step_data[key][i]))
                self.current_total_episode += 1

                if self.current_total_episode >= self.total_episode:
                    need_finish = True
            else:
                for key in step_data:
                    self.data[i][key].append(copy.copy(step_data[key][i]))

        return finished, need_finish

    def dump(self, save_path):
        self.all_trajectories["episode_lengths"] = self.episode_lengths
        self.all_trajectories["episode_rewards"] = self.episode_rewards

        trajectory_num = len(self.all_trajectories["obs"])
        for key in self.all_trajectories:
            assert len(self.all_trajectories[key]) == trajectory_num, (
                f"key: {key}, len: {len(self.all_trajectories[key])}, trajectory_num:"
                f" {trajectory_num}"
            )
        self.all_trajectories["env_info"] = {
            "agent_num": self.agent_num,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }

        with open(save_path, "wb") as f:
            pickle.dump(self.all_trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("data saved to: ", save_path)


# this creates dataset with larger size (about 4x) than GenDataWrapper_v1, thus need to be optimized
class GenDataWrapper(VecEnvWrapper):
    def __init__(self, env: BaseVecEnv, data_save_path: str, total_episode: int):
        assert env.env_is_wrapped(Monitor, indices=0)[0]

        super().__init__(env)

        self.data_save_path = data_save_path
        self.total_episode = total_episode
        self.pbar = None
        self.trajectory_data = None

    def reset(self, **kwargs):
        self.trajectory_data = TrajectoryData(
            self.env.parallel_env_num,
            self.total_episode,
            self.env.observation_space,
            self.env.action_space,
            self.env.agent_num,
        )

        returns = self.env.reset(**kwargs)
        if len(returns) == 2:
            obs = returns[0]
            info = returns[1]
        else:
            obs = returns
            info = {}
        reset_data = {"obs": obs, "info": info}
        self.trajectory_data.reset(reset_data)
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()
        self.pbar = tqdm(total=self.total_episode)

        return returns

    def step(self, action: ActType, *args, **kwargs):
        step_data = {}
        step_data["action"] = action
        obs, r, done, info = self.env.step(action, *args, **kwargs)
        step_data["obs"] = obs
        step_data["reward"] = r
        step_data["done"] = done
        step_data["info"] = info

        finished, need_finish = self.trajectory_data.step(step_data)
        self.pbar.update(len(finished))
        if need_finish:
            assert self.trajectory_data.current_total_episode == self.total_episode
        return obs, r, need_finish, info

    def close(self, **kwargs):
        self.pbar.refresh()
        self.pbar.close()
        average_length = np.mean(self.trajectory_data.episode_lengths)
        average_reward = np.mean(self.trajectory_data.episode_rewards)

        print(
            "collect total episode: {}".format(
                self.trajectory_data.current_total_episode
            )
        )
        print("average episode length: {}".format(average_length))
        print("average reward: {}".format(average_reward))

        self.trajectory_data.dump(self.data_save_path)
        return self.env.close(**kwargs)


class GenDataWrapper_v1(VecEnvWrapper):
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

        pickle.dump(
            self.data, open(self.data_save_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
        print("data saved to: ", self.data_save_path)
        return self.env.close(**kwargs)
