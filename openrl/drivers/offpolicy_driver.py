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
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from openrl.drivers.rl_driver import RLDriver
from openrl.utils.logger import Logger
from openrl.utils.type_aliases import MaybeCallback
from openrl.utils.util import _t2n


class OffPolicyDriver(RLDriver):
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
        super(OffPolicyDriver, self).__init__(
            config,
            trainer,
            buffer,
            agent,
            rank,
            world_size,
            client,
            logger,
            callback=callback,
        )

        self.buffer_minimal_size = int(config["cfg"].buffer_size * 0.2)
        self.epsilon_start = config["cfg"].epsilon_start
        self.epsilon_finish = config["cfg"].epsilon_finish
        self.epsilon_anneal_time = config["cfg"].epsilon_anneal_time

        self.algorithm_name = config["cfg"].algorithm_name
        self.var = config["cfg"].var
        self.obs_space = self.trainer.algo_module.obs_space
        self.act_space = self.trainer.algo_module.act_space
        self.var_step = self.var / self.num_env_steps

        if self.envs.parallel_env_num > 1:
            self.episode_steps = np.zeros((self.envs.parallel_env_num,))
        else:
            self.episode_steps = 0
        self.verbose_flag = False
        self.first_insert_buffer = True

    def _inner_loop(
        self,
    ) -> bool:
        """
        :return: True if training should continue, False if training should stop
        """
        rollout_infos = self.actor_rollout()

        if self.buffer.get_buffer_size() >= 0:
            train_infos = self.learner_update()
            self.buffer.after_update()
        else:
            train_infos = {"q_loss": 0}

        self.total_num_steps = (
            (self.episode + 1) * self.episode_length * self.n_rollout_threads
        )

        if self.episode % self.log_interval == 0:
            # rollout_infos can only be used when env is wrapped with VevMonitor
            # self.logger.log_info(rollout_infos, step=self.total_num_steps)
            # self.logger.log_info(train_infos, step=self.total_num_steps)
            pass

        return True

    def add2buffer(self, data):
        (
            obs,
            next_obs,
            rewards,
            dones,
            infos,
            q_values,
            actions,
            rnn_states,
        ) = data

        if self.algorithm_name == "DQN" or self.algorithm_name == "VDN":
            rnn_states[dones] = np.zeros(
                (dones.sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
        else:
            pass

        # todo add image obs
        if "Dict" in next_obs.__class__.__name__:
            for key in next_obs.keys():
                next_obs[key][dones] = np.zeros(
                    (dones.sum(), next_obs[key].shape[2]),
                    dtype=np.float32,
                )
        else:
            next_obs[dones] = np.zeros(
                (dones.sum(), next_obs.shape[2]),
                dtype=np.float32,
            )
            # pass

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones] = np.zeros((dones.sum(), 1), dtype=np.float32)

        rnn_states_critic = rnn_states
        action_log_probs = actions

        self.buffer.insert(
            obs,
            next_obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            q_values,
            rewards,
            masks,
        )

    def actor_rollout(self):
        self.callback.on_rollout_start()
        self.trainer.prep_rollout()

        obs = self.buffer.data.critic_obs[0]

        counter = 0
        ep_reward = 0
        for step in range(self.episode_length):
            if self.algorithm_name == "DQN" or self.algorithm_name == "VDN":
                q_values, actions, rnn_states = self.act(step)
                # print("step: ", step,
                #       "state: ", self.buffer.data.get_batch_data("next_policy_obs" if step != 0 else "policy_obs", step),
                #       "q_values: ", q_values,
                #       "actions: ", actions)
                extra_data = {
                    "q_values": q_values,
                    "step": step,
                    "buffer": self.buffer,
                }

                next_obs, rewards, dones, infos = self.envs.step(actions, extra_data)
                # print("rewards: ", rewards)

            elif self.algorithm_name == "DDPG":
                actions = self.act(step)
                extra_data = {
                    "step": step,
                    "buffer": self.buffer,
                }
                next_obs, rewards, dones, infos = self.envs.step(actions, extra_data)
                self.var -= self.var_step

                ep_reward += rewards

                q_values = np.zeros_like(actions)
                rnn_states = None

                # counter += 1
                if any(dones):
                    next_obs = np.array([infos[i]['final_observation'] for i in range(len(infos))])
                    # print("运行次数为：%d, 回报为：%.3f, 探索方差为：%.4f" % (counter, ep_reward, self.var))
                    # counter = 0
                    # ep_reward = 0

            all_dones = np.all(dones)
            if type(self.episode_steps) == int:
                if not all_dones:
                    self.episode_steps += 1
                else:
                    # print("steps: ", self.episode_steps)
                    self.episode_steps = 0
            else:
                done_index = list(np.where(dones == True)[0])
                self.episode_steps += 1
                for i in range(len(done_index)):
                    if self.episode_steps[done_index[i]] > 200:
                        self.verbose_flag = True
                    # print("steps: ", self.episode_steps[done_index[i]])
                    self.episode_steps[done_index[i]] = 0

            # Give access to local variables
            self.callback.update_locals(locals())
            if self.callback.on_step() is False:
                return {}, False

            data = (
                obs,
                next_obs,
                rewards,
                dones,
                infos,
                q_values,
                actions,
                rnn_states,
            )

            self.add2buffer(data)
            obs = next_obs

        batch_rew_infos = self.envs.batch_rewards(self.buffer)
        self.first_insert_buffer = False

        self.callback.on_rollout_end()

        if self.envs.use_monitor:
            statistics_info = self.envs.statistics(self.buffer)
            statistics_info.update(batch_rew_infos)
            return statistics_info
        else:
            return batch_rew_infos

    @torch.no_grad()
    def act(
        self,
        step: int,
    ):
        self.trainer.prep_rollout()

        if step != 0:
            step = step - 1

        if self.algorithm_name == "DQN" or self.algorithm_name == "VDN":
            (
                q_values,
                rnn_states,
            ) = self.trainer.algo_module.get_actions(
                self.buffer.data.get_batch_data(
                    "next_policy_obs" if step != 0 else "policy_obs", step
                ),
                np.concatenate(self.buffer.data.rnn_states[step]),
                np.concatenate(self.buffer.data.masks[step]),
            )

            q_values = np.array(np.split(_t2n(q_values), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

            epsilon = np.min(
                (
                    self.epsilon_finish
                    + (self.epsilon_start - self.epsilon_finish)
                    / self.epsilon_anneal_time
                    * (self.episode * self.episode_length + step),
                    self.epsilon_start,
                )

            )

            actions = np.expand_dims(q_values.argmax(axis=-1), axis=-1)

            if random.random() >= epsilon or self.first_insert_buffer:
                actions = np.random.randint(
                    low=0, high=self.envs.action_space.n, size=actions.shape)

            return (
                q_values,
                actions,
                rnn_states,
            )

        elif self.algorithm_name == "DDPG":
            actions = self.trainer.algo_module.get_actions(
                self.buffer.data.get_batch_data(
                    "next_policy_obs" if step != 0 else "policy_obs", step
                )
            ).numpy()

            actions = np.clip(
                np.random.normal(actions, self.var), self.act_space.low, self.act_space.high
            )

            actions = np.expand_dims(actions, -1)

            return actions


    def compute_returns(self):
        pass
