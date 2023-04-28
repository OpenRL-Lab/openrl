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
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from openrl.drivers.rl_driver import RLDriver
from openrl.utils.logger import Logger
from openrl.utils.util import _t2n


class OnPolicyDriver(RLDriver):
    def __init__(
        self,
        config: Dict[str, Any],
        trainer,
        buffer,
        rank: int = 0,
        world_size: int = 1,
        client=None,
        logger: Optional[Logger] = None,
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

    def _inner_loop(
        self,
    ) -> None:
        rollout_infos = self.actor_rollout()
        train_infos = self.learner_update()
        self.buffer.after_update()

        self.total_num_steps = (
            (self.episode + 1) * self.episode_length * self.n_rollout_threads
        )

        if self.episode % self.log_interval == 0:
            # rollout_infos can only be used when env is wrapped with VevMonitor
            self.logger.log_info(rollout_infos, step=self.total_num_steps)
            self.logger.log_info(train_infos, step=self.total_num_steps)

    def reset_and_buffer_init(self):
        returns = self.envs.reset()
        if isinstance(returns, tuple):
            assert (
                len(returns) == 2
            ), "length of env reset returns must be 2, but get {}".format(len(returns))
            obs, info = returns
        else:
            obs = returns

        self.buffer.init_buffer(obs.copy())

    def add2buffer(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones] = np.zeros(
            (dones.sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )

        rnn_states_critic[dones] = np.zeros(
            (dones.sum(), *self.buffer.data.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones] = np.zeros((dones.sum(), 1), dtype=np.float32)

        self.buffer.insert(
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    def actor_rollout(self):
        self.trainer.prep_rollout()
        import time

        for step in range(self.episode_length):
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.act(
                step
            )

            extra_data = {
                "values": values,
                "action_log_probs": action_log_probs,
                "step": step,
                "buffer": self.buffer,
            }

            obs, rewards, dones, infos = self.envs.step(actions, extra_data)

            data = (
                obs,
                rewards,
                dones,
                infos,
                values,
                actions,
                action_log_probs,
                rnn_states,
                rnn_states_critic,
            )

            self.add2buffer(data)

        batch_rew_infos = self.envs.batch_rewards(self.buffer)

        if self.envs.use_monitor:
            statistics_info = self.envs.statistics(self.buffer)
            statistics_info.update(batch_rew_infos)
            return statistics_info
        else:
            return batch_rew_infos

    def run(self) -> None:
        episodes = (
            int(self.num_env_steps)
            // self.episode_length
            // self.learner_n_rollout_threads
        )
        self.episodes = episodes

        self.reset_and_buffer_init()

        for episode in range(episodes):
            self.logger.info("Episode: {}/{}".format(episode, episodes))
            self.episode = episode
            self._inner_loop()

    def learner_update(self):
        if self.use_linear_lr_decay:
            self.trainer.algo_module.lr_decay(self.episode, self.episodes)

        self.compute_returns()

        self.trainer.prep_training()

        train_infos = self.trainer.train(self.buffer.data)

        return train_infos

    @torch.no_grad()
    def compute_returns(self):
        self.trainer.prep_rollout()

        next_values = self.trainer.algo_module.get_values(
            self.buffer.data.get_batch_data("critic_obs", -1),
            np.concatenate(self.buffer.data.rnn_states_critic[-1]),
            np.concatenate(self.buffer.data.masks[-1]),
        )

        next_values = np.array(
            np.split(_t2n(next_values), self.learner_n_rollout_threads)
        )
        if "critic" in self.trainer.algo_module.models and isinstance(
            self.trainer.algo_module.models["critic"], DistributedDataParallel
        ):
            value_normalizer = self.trainer.algo_module.models[
                "critic"
            ].module.value_normalizer
        elif "model" in self.trainer.algo_module.models and isinstance(
            self.trainer.algo_module.models["model"], DistributedDataParallel
        ):
            value_normalizer = self.trainer.algo_module.models["model"].value_normalizer
        else:
            value_normalizer = self.trainer.algo_module.get_critic_value_normalizer()
        self.buffer.compute_returns(next_values, value_normalizer)

    @torch.no_grad()
    def act(
        self,
        step: int,
    ):
        self.trainer.prep_rollout()

        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.algo_module.get_actions(
            self.buffer.data.get_batch_data("critic_obs", step),
            self.buffer.data.get_batch_data("policy_obs", step),
            np.concatenate(self.buffer.data.rnn_states[step]),
            np.concatenate(self.buffer.data.rnn_states_critic[step]),
            np.concatenate(self.buffer.data.masks[step]),
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        )
