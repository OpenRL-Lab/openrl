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
import io
import pathlib
from typing import Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch

from openrl.buffers.utils.obs_data import ObsData
from openrl.runners.common.base_agent import BaseAgent, SelfAgent
from openrl.utils.util import _t2n


class RLAgent(BaseAgent):
    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,
        env: Union[gym.Env, str] = None,
        run_dir: Optional[str] = None,
        env_num: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
    ) -> None:
        self.net = net
        self._cfg = net.cfg
        self._use_wandb = use_wandb
        self._use_tensorboard = not use_wandb and use_tensorboard

        if env is not None:
            self._env = env
        elif hasattr(net, "env") and net.env is not None:
            self._env = net.env
        else:
            raise ValueError("env is None")

        if env_num is not None:
            self.env_num = env_num
        else:
            self.env_num = self._env.parallel_env_num

        self._cfg.n_rollout_threads = self.env_num
        self._cfg.learner_n_rollout_threads = self._cfg.n_rollout_threads

        self.rank = rank
        self.world_size = world_size

        self.client = None
        self.agent_num = self._env.agent_num
        if run_dir is None:
            self.run_dir = self._cfg.run_dir
        else:
            self.run_dir = run_dir

        if self.run_dir is None:
            assert (not self._use_wandb) and (not self._use_tensorboard), (
                "run_dir must be set when using wandb or tensorboard. Please set"
                " run_dir in the config file or pass run_dir in the"
                " command line."
            )

        if self._cfg.experiment_name == "":
            self.exp_name = "rl"
        else:
            self.exp_name = self._cfg.experiment_name

    def train(self: SelfAgent, total_time_steps: int) -> None:
        raise NotImplementedError

    def act(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        assert self.net is not None, "net is None"
        observation = ObsData.prepare_input(observation)
        action, rnn_state = self.net.act(observation, deterministic=deterministic)

        action = np.array(np.split(_t2n(action), self.env_num))

        return action, rnn_state

    def set_env(
        self,
        env: Union[gym.Env, str] = None,
    ):
        self.net.reset()
        if env is not None:
            self._env = env
            self.env_num = env.parallel_env_num
        env.reset(seed=self._cfg.seed)
        self.net.reset(env)

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        if isinstance(path, str):
            path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.module, path / "module.pt")

    def load(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        if isinstance(path, str):
            path = pathlib.Path(path)

        assert path.exists(), f"{path} does not exist"

        if path.is_dir():
            path = path / "module.pt"

        assert path.exists(), f"{path} does not exist"

        if not torch.cuda.is_available():
            self.net.module = torch.load(path, map_location=torch.device("cpu"))
            self.net.module.device = torch.device("cpu")
            for key in self.net.module.models:
                self.net.module.models[key].tpdv = dict(
                    dtype=torch.float32, device=torch.device("cpu")
                )
        else:
            self.net.module = torch.load(path)

    def load_policy(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        self.net.load_policy(path)
