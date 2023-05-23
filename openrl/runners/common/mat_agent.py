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
from typing import Type

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.algorithms.mat import MATAlgorithm
from openrl.runners.common.base_agent import SelfAgent
from openrl.runners.common.ppo_agent import PPOAgent
from openrl.utils.logger import Logger


class MATAgent(PPOAgent):
    def train(
        self: SelfAgent,
        total_time_steps: int,
        train_algo_class: Type[BaseAlgorithm] = MATAlgorithm,
    ) -> None:
        logger = Logger(
            cfg=self._cfg,
            project_name="MATAgent",
            scenario_name=self._env.env_name,
            wandb_entity=self._cfg.wandb_entity,
            exp_name=self.exp_name,
            log_path=self.run_dir,
            use_wandb=self._use_wandb,
            use_tensorboard=self._use_tensorboard,
        )

        super(MATAgent, self).train(
            total_time_steps=total_time_steps,
            train_algo_class=train_algo_class,
            logger=logger,
        )
