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
from typing import Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.algorithms.dqn import DQNAlgorithm
from openrl.buffers import OffPolicyReplayBuffer as ReplayBuffer
from openrl.buffers.utils.obs_data import ObsData
from openrl.drivers.offpolicy_driver import OffPolicyDriver as Driver
from openrl.runners.common.base_agent import SelfAgent
from openrl.runners.common.rl_agent import RLAgent
from openrl.utils.logger import Logger
from openrl.utils.type_aliases import MaybeCallback
from openrl.utils.util import _t2n


class DQNAgent(RLAgent):
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
        super(DQNAgent, self).__init__(
            net, env, run_dir, env_num, rank, world_size, use_wandb, use_tensorboard
        )

    def train(
        self: SelfAgent,
        total_time_steps: int,
        callback: MaybeCallback = None,
        train_algo_class: Type[BaseAlgorithm] = DQNAlgorithm,
        logger: Optional[Logger] = None,
    ) -> None:
        self._cfg.num_env_steps = total_time_steps

        self.config = {
            "cfg": self._cfg,
            "num_agents": self.agent_num,
            "run_dir": self.run_dir,
            "envs": self._env,
            "device": self.net.device,
        }

        trainer = train_algo_class(
            cfg=self._cfg,
            init_module=self.net.module,
            device=self.net.device,
            agent_num=self.agent_num,
        )

        buffer = ReplayBuffer(
            self._cfg,
            self.agent_num,
            self._env.observation_space,
            self._env.action_space,
            data_client=None,
            episode_length=self._cfg.episode_length,
        )

        logger = Logger(
            cfg=self._cfg,
            project_name="DQNAgent",
            scenario_name=self._env.env_name,
            wandb_entity=self._cfg.wandb_entity,
            exp_name=self.exp_name,
            log_path=self.run_dir,
            use_wandb=self._use_wandb,
            use_tensorboard=self._use_tensorboard,
        )
        self._logger = logger

        total_time_steps, callback = self._setup_train(
            total_time_steps,
            callback,
            reset_num_time_steps=True,
            progress_bar=False,
        )

        driver = Driver(
            config=self.config,
            trainer=trainer,
            buffer=buffer,
            agent=self,
            client=self.client,
            rank=self.rank,
            world_size=self.world_size,
            logger=logger,
            callback=callback,
        )

        callback.on_training_start(locals(), globals())

        driver.run()

        callback.on_training_end()

    def act(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        assert self.net is not None, "net is None"
        observation = ObsData.prepare_input(observation)
        action, rnn_state = self.net.act(observation)

        action = np.array(np.split(_t2n(action), self.env_num))

        return action, rnn_state
