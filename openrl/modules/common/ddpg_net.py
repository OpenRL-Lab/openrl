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

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from openrl.configs.config import create_config_parser
from openrl.modules.common.base_net import BaseNet
from openrl.modules.ddpg_module import DDPGModule
from openrl.utils.util import set_seed


class DDPGNet(BaseNet):
    def __init__(
        self,
        env: Union[gym.Env, str],
        cfg=None,
        device: Union[torch.device, str] = "cpu",
        n_rollout_threads: int = 1,
        model_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if cfg is None:
            cfg_parser = create_config_parser()
            cfg = cfg_parser.parse_args()

        set_seed(cfg.seed)
        env.reset(seed=cfg.seed)

        cfg.n_rollout_threads = n_rollout_threads
        cfg.learner_n_rollout_threads = cfg.n_rollout_threads

        cfg.algorithm_name = "DDPG"

        if cfg.rnn_type == "gru":
            rnn_hidden_size = cfg.hidden_size
        elif cfg.rnn_type == "lstm":
            rnn_hidden_size = cfg.hidden_size * 2
        else:
            raise NotImplementedError(
                f"RNN type {cfg.rnn_type} has not been implemented."
            )
        cfg.rnn_hidden_size = rnn_hidden_size

        if isinstance(device, str):
            device = torch.device(device)

        self.module = DDPGModule(
            cfg=cfg,
            input_space=env.observation_space,
            act_space=env.action_space,
            device=device,
            rank=0,
            world_size=1,
            model_dict=model_dict,
        )

        self.cfg = cfg
        self.env = env
        self.device = device
        self.rnn_states_actor = None
        self.masks = None

    def act(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]], deterministic: bool
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        action = self.module.act(observation, deterministic).detach().numpy()

        return action

    def reset(self, env: Optional[gym.Env] = None) -> None:
        if env is not None:
            self.env = env
        self.first_reset = False
        self.rnn_states_actor, self.masks = self.module.init_rnn_states(
            rollout_num=self.env.parallel_env_num,
            agent_num=self.env.agent_num,
            rnn_layers=self.cfg.recurrent_N,
            hidden_size=self.cfg.rnn_hidden_size,
        )

    def load_policy(self, path: str) -> None:
        self.module.load_policy(path)
