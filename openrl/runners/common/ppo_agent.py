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
from typing import Optional, Union

import gym
import torch

from openrl.algorithms.ppo import PPOAlgorithm as TrainAlgo
from openrl.buffers import NormalReplayBuffer as ReplayBuffer
from openrl.drivers.onpolicy_driver import OnPolicyDriver as Driver
from openrl.runners.common.rl_agent import RLAgent
from openrl.runners.common.base_agent import SelfAgent
from openrl.utils.logger import Logger


class PPOAgent(RLAgent):
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
        super(PPOAgent, self).__init__(net, env, run_dir, env_num, rank, world_size, use_wandb, use_tensorboard)

    def train(self: SelfAgent, total_time_steps: int) -> None:
        self._cfg.num_env_steps = total_time_steps

        self.config = {
            "cfg": self._cfg,
            "num_agents": self.agent_num,
            "run_dir": self.run_dir,
            "envs": self._env,
            "device": self.net.device,
        }

        trainer = TrainAlgo(
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
        )

        logger = Logger(
            cfg=self._cfg,
            project_name="PPOAgent",
            scenario_name=self._env.env_name,
            wandb_entity=self._cfg.wandb_entity,
            exp_name=self.exp_name,
            log_path=self.run_dir,
            use_wandb=self._use_wandb,
            use_tensorboard=self._use_tensorboard,
        )
        driver = Driver(
            config=self.config,
            trainer=trainer,
            buffer=buffer,
            client=self.client,
            rank=self.rank,
            world_size=self.world_size,
            logger=logger,
        )
        driver.run()
