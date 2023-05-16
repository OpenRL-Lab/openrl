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
        super(OnPolicyDriver, self).__init__(
            config, trainer, buffer, rank, world_size, client, logger
        )

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
