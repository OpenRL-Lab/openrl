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

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from openrl.buffers.utils.obs_data import ObsData
from openrl.buffers.utils.util import (
    _cast,
    _cast_v3,
    _flatten,
    _flatten_v3,
    _shuffle_agent_grid,
    get_critic_obs,
    get_critic_obs_space,
    get_policy_obs,
    get_policy_obs_space,
    get_shape_from_act_space,
)


class ReplayData(object):
    def __init__(
        self,
        cfg,
        num_agents,
        obs_space,
        act_space,
        data_client=None,
        episode_length=None,
    ):
        if episode_length is None:
            episode_length = cfg.episode_length
        self.episode_length = episode_length

        self.n_rollout_threads = cfg.n_rollout_threads

        if hasattr(cfg, "rnn_hidden_size"):
            self.hidden_size = cfg.rnn_hidden_size
        else:
            self.hidden_size = cfg.hidden_size
        self.recurrent_N = cfg.recurrent_N
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self._use_gae = cfg.use_gae
        self._use_popart = cfg.use_popart
        self._use_valuenorm = cfg.use_valuenorm
        self._use_proper_time_limits = cfg.use_proper_time_limits

        self._mixed_obs = False  # for mixed observation

        policy_obs_shape = get_policy_obs_space(obs_space)
        critic_obs_shape = get_critic_obs_space(obs_space)

        # for mixed observation
        if "Dict" in policy_obs_shape.__class__.__name__:
            self._mixed_obs = True

            self.policy_obs = {}
            self.critic_obs = {}

            for key in policy_obs_shape:
                self.policy_obs[key] = np.zeros(
                    (
                        self.episode_length + 1,
                        self.n_rollout_threads,
                        num_agents,
                        *policy_obs_shape[key].shape,
                    ),
                    dtype=np.float32,
                )
            for key in critic_obs_shape:
                self.critic_obs[key] = np.zeros(
                    (
                        self.episode_length + 1,
                        self.n_rollout_threads,
                        num_agents,
                        *critic_obs_shape[key].shape,
                    ),
                    dtype=np.float32,
                )
            self.policy_obs = ObsData(self.policy_obs)
            self.critic_obs = ObsData(self.critic_obs)

        else:
            # deal with special attn format
            if type(policy_obs_shape[-1]) == list:
                policy_obs_shape[:1]

            if type(critic_obs_shape[-1]) == list:
                critic_obs_shape = critic_obs_shape[:1]

            self.critic_obs = np.zeros(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    *critic_obs_shape,
                ),
                dtype=np.float32,
            )
            self.policy_obs = np.zeros(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    *policy_obs_shape,
                ),
                dtype=np.float32,
            )

        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == "Discrete":
            self.action_masks = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.n,
                ),
                dtype=np.float32,
            )
        else:
            self.action_masks = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def get_batch_data(
        self,
        data_name: str,
        step: int,
    ):
        assert hasattr(self, data_name)
        data = getattr(self, data_name)
        if data is None:
            return None

        if isinstance(data, ObsData):
            return data.step_batch(step)
        else:
            return np.concatenate(data[step])

    def all_batch_data(self, data_name: str, min=None, max=None):
        assert hasattr(self, data_name)
        data = getattr(self, data_name)

        if isinstance(data, ObsData):
            return data.all_batch(min, max)
        else:
            return data[min:max].reshape((-1, *data.shape[3:]))

    def dict_insert(self, data):
        if self._mixed_obs:
            for key in self.critic_obs.keys():
                self.critic_obs[key][self.step + 1] = data["critic_obs"][key].copy()
            for key in self.policy_obs.keys():
                self.policy_obs[key][self.step + 1] = data["policy_obs"][key].copy()
        else:
            self.critic_obs[self.step + 1] = data["critic_obs"].copy()
            self.policy_obs[self.step + 1] = data["policy_obs"].copy()

        if "rnn_states" in data:
            self.rnn_states[self.step + 1] = data["rnn_states"].copy()
        if "rnn_states_critic" in data:
            self.rnn_states_critic[self.step + 1] = data["rnn_states_critic"].copy()
        if "actions" in data:
            self.actions[self.step] = data["actions"].copy()
        if "action_log_probs" in data:
            self.action_log_probs[self.step] = data["action_log_probs"].copy()

        if "value_preds" in data:
            self.value_preds[self.step] = data["value_preds"].copy()
        if "rewards" in data:
            self.rewards[self.step] = data["rewards"].copy()
        if "masks" in data:
            self.masks[self.step + 1] = data["masks"].copy()

        if "bad_masks" in data:
            self.bad_masks[self.step + 1] = data["bad_masks"].copy()
        if "active_masks" in data:
            self.active_masks[self.step + 1] = data["active_masks"].copy()
        if "action_masks" in data:
            self.action_masks[self.step + 1] = data["action_masks"].copy()

        self.step = (self.step + 1) % self.episode_length

    def insert(
        self,
        raw_obs,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        action_masks=None,
    ):
        critic_obs = get_critic_obs(raw_obs)
        policy_obs = get_policy_obs(raw_obs)
        if self._mixed_obs:
            for key in self.critic_obs.keys():
                self.critic_obs[key][self.step + 1] = critic_obs[key].copy()
            for key in self.policy_obs.keys():
                self.policy_obs[key][self.step + 1] = policy_obs[key].copy()
        else:
            self.critic_obs[self.step + 1] = critic_obs.copy()
            self.policy_obs[self.step + 1] = policy_obs.copy()
        if rnn_states is not None:
            self.rnn_states[self.step + 1] = rnn_states.copy()
        if rnn_states_critic is not None:
            self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if action_masks is not None:
            self.action_masks[self.step + 1] = action_masks.copy()
        self.step = (self.step + 1) % self.episode_length

    def init_buffer(self, raw_obs, action_masks=None):
        critic_obs = get_critic_obs(raw_obs)
        policy_obs = get_policy_obs(raw_obs)
        if self._mixed_obs:
            for key in self.critic_obs.keys():
                self.critic_obs[key][0] = critic_obs[key].copy()
            for key in self.policy_obs.keys():
                self.policy_obs[key][0] = policy_obs[key].copy()
        else:
            self.critic_obs[0] = critic_obs.copy()
            self.policy_obs[0] = policy_obs.copy()
        if action_masks is not None and self.action_masks is not None:
            self.action_masks[0] = action_masks

    def after_update(self):
        assert self.step == 0, "step:{} episode:{}".format(
            self.step, self.episode_length
        )
        if self._mixed_obs:
            for key in self.critic_obs.keys():
                self.critic_obs[key][0] = self.critic_obs[key][-1].copy()
            for key in self.policy_obs.keys():
                self.policy_obs[key][0] = self.policy_obs[key][-1].copy()
        else:
            self.critic_obs[0] = self.critic_obs[-1].copy()
            self.policy_obs[0] = self.policy_obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.action_masks is not None:
            self.action_masks[0] = self.action_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if (
                        self._use_popart or self._use_valuenorm
                    ) and value_normalizer is not None:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def recurrent_generator_v3(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length >= data_chunk_length, (
            "PPO requires the nfumber of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(
                n_rollout_threads, episode_length, data_chunk_length
            )
        )

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        critic_obs = _cast_v3(self.critic_obs[:-1])
        policy_obs = _cast_v3(self.policy_obs[:-1])

        actions = _cast_v3(self.actions)
        action_log_probs = _cast_v3(self.action_log_probs)
        advantages = _cast_v3(advantages)
        value_preds = _cast_v3(self.value_preds[:-1])
        returns = _cast_v3(self.returns[:-1])
        masks = _cast_v3(self.masks[:-1])
        active_masks = _cast_v3(self.active_masks[:-1])

        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.rnn_states.shape[2:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        if self.action_masks is not None:
            action_masks = _cast_v3(self.action_masks[:-1])

        for indices in sampler:
            critic_obs_batch = []
            policy_obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            action_masks_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                # [L, agent_num, Dim]
                critic_obs_batch.append(critic_obs[ind : ind + data_chunk_length])
                policy_obs_batch.append(policy_obs[ind : ind + data_chunk_length])

                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.action_masks is not None:
                    action_masks_batch.append(
                        action_masks[ind : ind + data_chunk_length]
                    )
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind : ind + data_chunk_length]
                )
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                # [1,agent_num, Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, agent_num, Dim)

            critic_obs_batch = np.stack(critic_obs_batch, axis=1)
            policy_obs_batch = np.stack(policy_obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.action_masks is not None:
                action_masks_batch = np.stack(action_masks_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, agent_num, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N * num_agents, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N * num_agents, *self.rnn_states_critic.shape[3:]
            )
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)

            critic_obs_batch = _flatten_v3(L, N, num_agents, critic_obs_batch)
            policy_obs_batch = _flatten_v3(L, N, num_agents, policy_obs_batch)
            actions_batch = _flatten_v3(L, N, num_agents, actions_batch)
            if self.action_masks is not None:
                action_masks_batch = _flatten_v3(L, N, num_agents, action_masks_batch)
            else:
                action_masks_batch = None
            value_preds_batch = _flatten_v3(L, N, num_agents, value_preds_batch)
            return_batch = _flatten_v3(L, N, num_agents, return_batch)
            masks_batch = _flatten_v3(L, N, num_agents, masks_batch)
            active_masks_batch = _flatten_v3(L, N, num_agents, active_masks_batch)
            old_action_log_probs_batch = _flatten_v3(
                L, N, num_agents, old_action_log_probs_batch
            )
            adv_targ = _flatten_v3(L, N, num_agents, adv_targ)
            yield critic_obs_batch, policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch

    def feed_forward_generator(
        self,
        advantages,
        num_mini_batch=None,
        mini_batch_size=None,
        critic_obs_process_func=None,
    ):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert (
                batch_size >= num_mini_batch
            ), (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    n_rollout_threads,
                    episode_length,
                    num_agents,
                    n_rollout_threads * episode_length * num_agents,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        if self._mixed_obs:
            critic_obs = {}
            policy_obs = {}
            for key in self.critic_obs.keys():
                critic_obs[key] = self.critic_obs[key][:-1].reshape(
                    -1, *self.critic_obs[key].shape[3:]
                )
            for key in self.policy_obs.keys():
                policy_obs[key] = self.policy_obs[key][:-1].reshape(
                    -1, *self.policy_obs[key].shape[3:]
                )
        else:
            critic_obs = self.critic_obs[:-1].reshape(-1, *self.critic_obs.shape[3:])
            policy_obs = self.policy_obs[:-1].reshape(-1, *self.policy_obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.action_masks is not None:
            action_masks = self.action_masks[:-1].reshape(
                -1, self.action_masks.shape[-1]
            )
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1]
        )
        if advantages is not None:
            advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            if self._mixed_obs:
                critic_obs_batch = {}
                policy_obs_batch = {}
                for key in critic_obs.keys():
                    critic_obs_batch[key] = critic_obs[key][indices]
                for key in policy_obs.keys():
                    policy_obs_batch[key] = policy_obs[key][indices]
            else:
                critic_obs_batch = critic_obs[indices]
                policy_obs_batch = policy_obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.action_masks is not None:
                action_masks_batch = action_masks[indices]
            else:
                action_masks_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            if critic_obs_process_func is not None:
                critic_obs_batch = critic_obs_process_func(critic_obs_batch)

            yield critic_obs_batch, policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch

    def feed_forward_critic_obs_generator(
        self,
        advantages,
        num_mini_batch=None,
        mini_batch_size=None,
        critic_obs_process_func=None,
    ):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert (
                batch_size >= num_mini_batch
            ), (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    n_rollout_threads,
                    episode_length,
                    num_agents,
                    n_rollout_threads * episode_length,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        if self._mixed_obs:
            critic_obs = {}
            for key in self.critic_obs.keys():
                critic_obs[key] = self.critic_obs[key][:-1].reshape(
                    -1, *self.critic_obs[key].shape[3:]
                )
        else:
            critic_obs = self.critic_obs[:-1, :, 0].reshape(
                -1, *self.critic_obs.shape[3:]
            )  # [T*N,Dim]

        actions = self.actions[:, :, 0].reshape(-1, self.actions.shape[-1])

        for indices in sampler:
            # T is episode length, N is rollout number, M is agent number, dim is dimension
            # critic_obs size [T+1 N M Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            if self._mixed_obs:
                critic_obs_batch = {}
                for key in critic_obs.keys():
                    critic_obs_batch[key] = critic_obs[key][indices]
            else:
                critic_obs_batch = critic_obs[indices]

            actions_batch = actions[indices]

            if critic_obs_process_func is not None:
                critic_obs_batch = critic_obs_process_func(critic_obs_batch)

            yield critic_obs_batch, None, None, None, actions_batch, None, None, None, None, None, None, None

    def feed_forward_generator_transformer(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert (
                batch_size >= num_mini_batch
            ), (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    n_rollout_threads,
                    episode_length,
                    n_rollout_threads * episode_length,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        critic_obs = self.critic_obs[:-1].reshape(-1, *self.critic_obs.shape[2:])
        critic_obs = critic_obs[rows, cols]
        policy_obs = self.policy_obs[:-1].reshape(-1, *self.policy_obs.shape[2:])
        policy_obs = policy_obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])

        actions = actions[rows, cols]

        if self.action_masks is not None:
            action_masks = self.action_masks[:-1].reshape(
                -1, *self.action_masks.shape[2:]
            )
            action_masks = action_masks[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(
            -1, *self.action_log_probs.shape[2:]
        )
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            critic_obs_batch = critic_obs[indices].reshape(-1, *critic_obs.shape[2:])
            policy_obs_batch = policy_obs[indices].reshape(-1, *policy_obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(
                -1, *rnn_states_critic.shape[2:]
            )
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.action_masks is not None:
                action_masks_batch = action_masks[indices].reshape(
                    -1, *action_masks.shape[2:]
                )
            else:
                action_masks_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(
                -1, *active_masks.shape[2:]
            )
            old_action_log_probs_batch = action_log_probs[indices].reshape(
                -1, *action_log_probs.shape[2:]
            )
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield critic_obs_batch, policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(
                n_rollout_threads, num_agents, num_mini_batch
            )
        )
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        if self._mixed_obs:
            critic_obs = {}
            policy_obs = {}
            for key in self.critic_obs.keys():
                critic_obs[key] = self.critic_obs[key].reshape(
                    -1, batch_size, *self.critic_obs[key].shape[3:]
                )
            for key in self.policy_obs.keys():
                policy_obs[key] = self.policy_obs[key].reshape(
                    -1, batch_size, *self.policy_obs[key].shape[3:]
                )
        else:
            critic_obs = self.critic_obs.reshape(
                -1, batch_size, *self.critic_obs.shape[3:]
            )
            policy_obs = self.policy_obs.reshape(
                -1, batch_size, *self.policy_obs.shape[3:]
            )
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(
            -1, batch_size, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.action_masks is not None:
            action_masks = self.action_masks.reshape(
                -1, batch_size, self.action_masks.shape[-1]
            )
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1]
        )
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            if self._mixed_obs:
                critic_obs_batch = defaultdict(list)
                policy_obs_batch = defaultdict(list)
            else:
                critic_obs_batch = []
                policy_obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            action_masks_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self._mixed_obs:
                    for key in critic_obs.keys():
                        critic_obs_batch[key].append(critic_obs[key][:-1, ind])
                    for key in policy_obs.keys():
                        policy_obs_batch[key].append(policy_obs[key][:-1, ind])
                else:
                    critic_obs_batch.append(critic_obs[:-1, ind])
                    policy_obs_batch.append(policy_obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.action_masks is not None:
                    action_masks_batch.append(action_masks[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            if self._mixed_obs:
                for key in critic_obs_batch.keys():
                    critic_obs_batch[key] = np.stack(critic_obs_batch[key], 1)
                for key in policy_obs_batch.keys():
                    policy_obs_batch[key] = np.stack(policy_obs_batch[key], 1)
            else:
                critic_obs_batch = np.stack(critic_obs_batch, 1)
                policy_obs_batch = np.stack(policy_obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.action_masks is not None:
                action_masks_batch = np.stack(action_masks_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            if self._mixed_obs:
                for key in critic_obs_batch.keys():
                    critic_obs_batch[key] = _flatten(T, N, critic_obs_batch[key])
                for key in policy_obs_batch.keys():
                    policy_obs_batch[key] = _flatten(T, N, policy_obs_batch[key])
            else:
                critic_obs_batch = _flatten(T, N, critic_obs_batch)
                policy_obs_batch = _flatten(T, N, policy_obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.action_masks is not None:
                action_masks_batch = _flatten(T, N, action_masks_batch)
            else:
                action_masks_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield critic_obs_batch, policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch

    def recurrent_generator_v2(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert (
                batch_size >= num_mini_batch
            ), (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    n_rollout_threads,
                    episode_length,
                    n_rollout_threads * episode_length,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        # keep (num_agent, dim)
        critic_obs = self.critic_obs[:-1].reshape(-1, *self.critic_obs.shape[2:])

        policy_obs = self.policy_obs[:-1].reshape(-1, *self.policy_obs.shape[2:])

        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])

        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )

        actions = self.actions.reshape(-1, *self.actions.shape[2:])

        if self.action_masks is not None:
            action_masks = self.action_masks[:-1].reshape(
                -1, *self.action_masks.shape[2:]
            )

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])

        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])

        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])

        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])

        action_log_probs = self.action_log_probs.reshape(
            -1, *self.action_log_probs.shape[2:]
        )

        advantages = advantages.reshape(-1, *advantages.shape[2:])

        shuffle = False
        if shuffle:
            rows, cols = _shuffle_agent_grid(batch_size, num_agents)

            if self.action_masks is not None:
                action_masks = action_masks[rows, cols]
            critic_obs = critic_obs[rows, cols]
            policy_obs = policy_obs[rows, cols]
            rnn_states = rnn_states[rows, cols]
            rnn_states_critic = rnn_states_critic[rows, cols]
            actions = actions[rows, cols]
            value_preds = value_preds[rows, cols]
            returns = returns[rows, cols]
            masks = masks[rows, cols]
            active_masks = active_masks[rows, cols]
            action_log_probs = action_log_probs[rows, cols]
            advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            critic_obs_batch = critic_obs[indices].reshape(-1, *critic_obs.shape[2:])
            policy_obs_batch = policy_obs[indices].reshape(-1, *policy_obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(
                -1, *rnn_states_critic.shape[2:]
            )
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.action_masks is not None:
                action_masks_batch = action_masks[indices].reshape(
                    -1, *action_masks.shape[2:]
                )
            else:
                action_masks_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(
                -1, *active_masks.shape[2:]
            )
            old_action_log_probs_batch = action_log_probs[indices].reshape(
                -1, *action_log_probs.shape[2:]
            )
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])
            yield critic_obs_batch, policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]

        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode"
            " length ({}) to be greater than or equal to the number of data chunk"
            " length ({}).".format(
                n_rollout_threads, num_agents, episode_length, data_chunk_length
            )
        )

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        if self._mixed_obs:
            critic_obs = {}
            policy_obs = {}
            for key in self.critic_obs.keys():
                if len(self.critic_obs[key].shape) == 6:
                    critic_obs[key] = (
                        self.critic_obs[key][:-1]
                        .transpose(1, 2, 0, 3, 4, 5)
                        .reshape(-1, *self.critic_obs[key].shape[3:])
                    )
                elif len(self.critic_obs[key].shape) == 5:
                    critic_obs[key] = (
                        self.critic_obs[key][:-1]
                        .transpose(1, 2, 0, 3, 4)
                        .reshape(-1, *self.critic_obs[key].shape[3:])
                    )
                else:
                    critic_obs[key] = _cast(self.critic_obs[key][:-1])

            for key in self.policy_obs.keys():
                if len(self.policy_obs[key].shape) == 6:
                    policy_obs[key] = (
                        self.policy_obs[key][:-1]
                        .transpose(1, 2, 0, 3, 4, 5)
                        .reshape(-1, *self.policy_obs[key].shape[3:])
                    )
                elif len(self.policy_obs[key].shape) == 5:
                    policy_obs[key] = (
                        self.policy_obs[key][:-1]
                        .transpose(1, 2, 0, 3, 4)
                        .reshape(-1, *self.policy_obs[key].shape[3:])
                    )
                else:
                    policy_obs[key] = _cast(self.policy_obs[key][:-1])
        else:
            if len(self.critic_obs.shape) > 4:
                critic_obs = (
                    self.critic_obs[:-1]
                    .transpose(1, 2, 0, 3, 4, 5)
                    .reshape(-1, *self.critic_obs.shape[3:])
                )
                policy_obs = (
                    self.policy_obs[:-1]
                    .transpose(1, 2, 0, 3, 4, 5)
                    .reshape(-1, *self.policy_obs.shape[3:])
                )
            else:
                critic_obs = _cast(self.critic_obs[:-1])
                policy_obs = _cast(self.policy_obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])

        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states.shape[3:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[3:])
        )

        if self.action_masks is not None:
            action_masks = _cast(self.action_masks[:-1])

        for indices in sampler:
            if self._mixed_obs:
                critic_obs_batch = defaultdict(list)
                policy_obs_batch = defaultdict(list)
            else:
                critic_obs_batch = []
                policy_obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            action_masks_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in critic_obs.keys():
                        critic_obs_batch[key].append(
                            critic_obs[key][ind : ind + data_chunk_length]
                        )
                    for key in policy_obs.keys():
                        policy_obs_batch[key].append(
                            policy_obs[key][ind : ind + data_chunk_length]
                        )
                else:
                    critic_obs_batch.append(critic_obs[ind : ind + data_chunk_length])
                    policy_obs_batch.append(policy_obs[ind : ind + data_chunk_length])

                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.action_masks is not None:
                    action_masks_batch.append(
                        action_masks[ind : ind + data_chunk_length]
                    )
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind : ind + data_chunk_length]
                )
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            if self._mixed_obs:
                for key in critic_obs_batch.keys():
                    critic_obs_batch[key] = np.stack(critic_obs_batch[key], axis=1)
                for key in policy_obs_batch.keys():
                    policy_obs_batch[key] = np.stack(policy_obs_batch[key], axis=1)
            else:
                critic_obs_batch = np.stack(critic_obs_batch, axis=1)
                policy_obs_batch = np.stack(policy_obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.action_masks is not None:
                action_masks_batch = np.stack(action_masks_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                for key in critic_obs_batch.keys():
                    critic_obs_batch[key] = _flatten(L, N, critic_obs_batch[key])
                for key in policy_obs_batch.keys():
                    policy_obs_batch[key] = _flatten(L, N, policy_obs_batch[key])
            else:
                critic_obs_batch = _flatten(L, N, critic_obs_batch)
                policy_obs_batch = _flatten(L, N, policy_obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.action_masks is not None:
                action_masks_batch = _flatten(L, N, action_masks_batch)
            else:
                action_masks_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield critic_obs_batch, policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch
