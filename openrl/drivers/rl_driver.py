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
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from openrl.drivers.base_driver import BaseDriver
from openrl.envs.vec_env.utils.util import prepare_action_masks
from openrl.utils.logger import Logger
from openrl.utils.type_aliases import MaybeCallback


class RLDriver(BaseDriver, ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        trainer,
        buffer,
        agent,
        rank: int = 0,
        world_size: int = 1,
        client=None,
        logger: Optional[Logger] = None,
        callback: MaybeCallback = None,
    ) -> None:
        self.trainer = trainer
        self.buffer = buffer
        self.learner_episode = -1
        self.actor_id = 0
        self.weight_ids = [0]
        self.world_size = world_size
        self.logger = logger
        cfg = config["cfg"]
        self.program_type = cfg.program_type
        self.envs = config["envs"]
        self.device = config["device"]
        self.callback = callback
        self.agent = agent

        assert not (
            self.program_type != "actor" and self.world_size is None
        ), "world size can not be none, get {}".format(world_size)

        self.num_agents = config["num_agents"]
        assert isinstance(rank, int), "rank must be int, but get {}".format(rank)
        self.rank = rank

        # for distributed learning
        assert not (
            world_size is None and self.program_type == "learner"
        ), "world_size must be int, but get {}".format(world_size)

        # parameters
        self.env_name = cfg.env_name
        self.algorithm_name = cfg.algorithm_name
        self.experiment_name = cfg.experiment_name

        self.num_env_steps = cfg.num_env_steps
        self.episode_length = cfg.episode_length
        self.n_rollout_threads = cfg.n_rollout_threads
        self.learner_n_rollout_threads = cfg.learner_n_rollout_threads
        self.n_eval_rollout_threads = cfg.n_eval_rollout_threads
        self.n_render_rollout_threads = cfg.n_render_rollout_threads
        self.use_linear_lr_decay = cfg.use_linear_lr_decay
        self.hidden_size = cfg.hidden_size
        self.use_wandb = not cfg.disable_wandb
        self.use_single_network = cfg.use_single_network
        self.use_render = cfg.use_render
        self.use_transmit = cfg.use_transmit
        self.recurrent_N = cfg.recurrent_N
        self.only_eval = cfg.only_eval
        self.save_interval = cfg.save_interval
        self.use_eval = cfg.use_eval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval

        self.distributed_type = cfg.distributed_type

        self.actor_num = cfg.actor_num

        if self.distributed_type == "async" and self.program_type == "whole":
            print("can't use async mode when program_type is whole!")
            exit()

        if self.program_type in ["whole", "local"]:
            assert self.actor_num == 1, (
                "when running actor and learner the same time, the actor number should"
                " be 1, but received {}".format(self.actor_num)
            )
        # dir
        self.model_dir = cfg.model_dir
        if hasattr(cfg, "save_dir"):
            self.save_dir = cfg.save_dir

        self.cfg = cfg

    @abstractmethod
    def _inner_loop(self) -> bool:
        """
        :return: True if training should continue, False if training should stop
        """
        raise NotImplementedError

    def reset_and_buffer_init(self):
        returns = self.envs.reset()
        if isinstance(returns, tuple):
            assert (
                len(returns) == 2
            ), "length of env reset returns must be 2, but get {}".format(len(returns))
            obs, info = returns
        else:
            obs = returns
            info = None
        action_masks = prepare_action_masks(
            info, agent_num=self.num_agents, as_batch=False
        )
        self.buffer.init_buffer(obs.copy(), action_masks=action_masks)

    @abstractmethod
    def add2buffer(self, data):
        raise NotImplementedError

    @abstractmethod
    def actor_rollout(self):
        raise NotImplementedError

    def run(self) -> None:
        episodes = (
            int(self.num_env_steps)
            // self.episode_length
            // self.learner_n_rollout_threads
        )
        self.episodes = episodes

        self.reset_and_buffer_init()
        self.real_step = 0
        for episode in range(episodes):
            self.logger.info("Episode: {}/{}".format(episode, episodes))
            self.episode = episode
            continue_training = self._inner_loop()
            if not continue_training:
                break

    def learner_update(self):
        if self.use_linear_lr_decay:
            self.trainer.algo_module.lr_decay(self.episode, self.episodes)

        self.compute_returns()

        self.trainer.prep_training()

        train_infos = self.trainer.train(self.buffer.data)

        return train_infos

    @abstractmethod
    def compute_returns(self):
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        step: int,
    ):
        raise NotImplementedError
