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
from typing import Optional, Type, Union

import gym
import torch

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.algorithms.behavior_cloning import BCAlgorithm
from openrl.drivers.base_driver import BaseDriver
from openrl.drivers.offline_driver import OfflineDriver as Driver
from openrl.modules.common import BaseNet
from openrl.runners.common.base_agent import SelfAgent
from openrl.runners.common.ppo_agent import PPOAgent
from openrl.utils.logger import Logger
from openrl.utils.type_aliases import MaybeCallback


class BCAgent(PPOAgent):
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
        project_name: str = "GAILAgent",
    ) -> None:
        super(BCAgent, self).__init__(
            net,
            env,
            run_dir,
            env_num,
            rank,
            world_size,
            use_wandb,
            use_tensorboard,
            project_name=project_name,
        )

    def train(
        self: SelfAgent,
        total_time_steps: int,
        callback: MaybeCallback = None,
        train_algo_class: Type[BaseAlgorithm] = BCAlgorithm,
        logger: Optional[Logger] = None,
        DriverClass: Type[BaseDriver] = Driver,
    ) -> None:
        super().train(
            total_time_steps,
            callback,
            train_algo_class,
            logger,
            DriverClass=DriverClass,
        )
