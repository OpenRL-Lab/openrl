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
import time
from abc import abstractmethod
from typing import Optional, Tuple, Union

import gym
import torch

from openrl.modules.common import BaseNet
from openrl.runners.common.base_agent import BaseAgent, SelfAgent
from openrl.utils.callbacks import CallbackFactory
from openrl.utils.callbacks.callbacks import BaseCallback, CallbackList, ConvertCallback
from openrl.utils.callbacks.processbar_callback import ProgressBarCallback
from openrl.utils.type_aliases import MaybeCallback


class RLAgent(BaseAgent):
    def __init__(
        self,
        net: Optional[Union[torch.nn.Module, BaseNet]] = None,
        env: Union[gym.Env, str] = None,
        run_dir: Optional[str] = None,
        env_num: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project_name: str = "RLAgent",
    ) -> None:
        self.net = net
        if self.net is not None:
            self.net.reset()
        self._cfg = net.cfg
        self._use_wandb = use_wandb
        self._use_tensorboard = not use_wandb and use_tensorboard
        self.project_name = project_name

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

        # current number of timesteps
        self.num_time_steps = 0
        self._episode_num = 0
        self._total_time_steps = 0

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

    @abstractmethod
    def train(
        self: SelfAgent,
        total_time_steps: int,
        callback: MaybeCallback = None,
    ) -> None:
        raise NotImplementedError

    def _setup_train(
        self,
        total_time_steps: int,
        callback: MaybeCallback = None,
        reset_num_time_steps: bool = True,
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_time_steps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_time_steps: Whether to reset or not the ``num_time_steps`` attribute

        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total time_steps and callback(s)
        """
        self.start_time = time.time_ns()

        if reset_num_time_steps:
            self.num_time_steps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_time_steps += self.num_time_steps
        self._total_time_steps = total_time_steps

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_time_steps, callback

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        if self._cfg.callbacks:
            cfg_callback = CallbackFactory.get_callbacks(self._cfg.callbacks)
            callback = CallbackList([callback, cfg_callback])
        callback.init_callback(self)
        return callback

    @abstractmethod
    def act(self, **kwargs) -> None:
        raise NotImplementedError

    def set_env(
        self,
        env: Union[gym.Env, str],
    ):
        self.net.reset()

        if env is not None:
            self._env = env
            self.env_num = env.parallel_env_num
            self.agent_num = env.agent_num
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
        self.net.reset()

    def load_policy(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        self.net.load_policy(path)
