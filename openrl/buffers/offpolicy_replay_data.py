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

import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from openrl.buffers.replay_data import ReplayData
from openrl.buffers.utils.util import (
    get_critic_obs,
    get_critic_obs_space,
    get_policy_obs,
    get_policy_obs_space,
    get_shape_from_act_space,
)


class OffPolicyReplayData(ReplayData):
    def __init__(
        self,
        cfg,
        num_agents,
        obs_space,
        act_space,
        data_client=None,
        episode_length=None,
    ):
        super(OffPolicyReplayData, self).__init__(
            cfg,
            num_agents,
            obs_space,
            act_space,
            data_client,
            episode_length,
        )

        if act_space.__class__.__name__ == "Box":
            self.act_space = act_space.shape[0]
        elif act_space.__class__.__name__ == "Discrete":
            self.act_space = act_space.n

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )

        self.value_preds = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.act_space,
            ),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        policy_obs_shape = get_policy_obs_space(obs_space)
        critic_obs_shape = get_critic_obs_space(obs_space)
        self.next_policy_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *policy_obs_shape,
            ),
            dtype=np.float32,
        )
        self.next_critic_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *critic_obs_shape,
            ),
            dtype=np.float32,
        )
        self.first_insert_flag = True

    def dict_insert(self, data):
        if self._mixed_obs:
            for key in self.critic_obs.keys():
                self.critic_obs[key][self.step + 1] = data["critic_obs"][key].copy()
            for key in self.policy_obs.keys():
                self.policy_obs[key][self.step + 1] = data["policy_obs"][key].copy()
            for key in self.next_policy_obs.keys():
                self.next_policy_obs[key][self.step + 1] = data["next_policy_obs"][
                    key
                ].copy()
            for key in self.next_critic_obs.keys():
                self.next_critic_obs[key][self.step + 1] = data["next_critic_obs"][
                    key
                ].copy()
        else:
            self.critic_obs[self.step + 1] = data["critic_obs"].copy()
            self.policy_obs[self.step + 1] = data["policy_obs"].copy()
            self.next_policy_obs[self.step + 1] = data["next_policy_obs"].copy()
            self.next_critic_obs[self.step + 1] = data["next_critic_obs"].copy()

        if "rnn_states" in data:
            self.rnn_states[self.step + 1] = data["rnn_states"].copy()
        if "rnn_states_critic" in data:
            self.rnn_states_critic[self.step + 1] = data["rnn_states_critic"].copy()
        if "actions" in data:
            self.actions[self.step + 1] = data["actions"].copy()
        if "action_log_probs" in data:
            self.action_log_probs[self.step] = data["action_log_probs"].copy()

        if "value_preds" in data:
            self.value_preds[self.step] = data["value_preds"].copy()
        if "rewards" in data:
            self.rewards[self.step + 1] = data["rewards"].copy()
        if "masks" in data:
            self.masks[self.step + 1] = data["masks"].copy()

        if "bad_masks" in data:
            self.bad_masks[self.step + 1] = data["bad_masks"].copy()
        if "active_masks" in data:
            self.active_masks[self.step + 1] = data["active_masks"].copy()
        if "action_masks" in data:
            self.action_masks[self.step + 1] = data["action_masks"].copy()

        if (self.step + 1) % self.episode_length != 0:
            self.first_insert_flag = False
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

    def insert(
        self,
        raw_obs,
        # next_raw_obs,
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
            for key in self.next_critic_obs.keys():
                self.next_critic_obs[key][self.step] = critic_obs[key].copy()
            for key in self.next_policy_obs.keys():
                self.next_policy_obs[key][self.step] = policy_obs[key].copy()
        else:
            self.critic_obs[self.step + 1] = critic_obs.copy()
            self.policy_obs[self.step + 1] = policy_obs.copy()

            self.next_critic_obs[self.step] = critic_obs.copy()
            self.next_policy_obs[self.step] = policy_obs.copy()
        if rnn_states is not None:
            self.rnn_states[self.step + 1] = rnn_states.copy()
        if rnn_states_critic is not None:
            self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        # self.rewards[self.step + 1] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.rewards[self.step] = rewards.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if action_masks is not None:
            self.action_masks[self.step + 1] = action_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self, next_value, value_normalizer=None):
        pass

    def after_update(self):
        assert self.step == 0, "step:{} episode:{}".format(
            self.step, self.episode_length
        )
        if self._mixed_obs:
            for key in self.critic_obs.keys():
                self.critic_obs[key][0] = self.critic_obs[key][-1].copy()
                self.next_critic_obs[key][0] = self.next_critic_obs[key][-1].copy()

            for key in self.policy_obs.keys():
                self.policy_obs[key][0] = self.policy_obs[key][-1].copy()
                self.next_policy_obs[key][0] = self.next_policy_obs[key][-1].copy()
        else:
            self.critic_obs[0] = self.critic_obs[-1].copy()
            self.next_critic_obs[0] = self.next_critic_obs[-1].copy()
            self.policy_obs[0] = self.policy_obs[-1].copy()
            self.next_policy_obs[0] = self.next_policy_obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.actions[0] = self.actions[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.action_masks is not None:
            self.action_masks[0] = self.action_masks[-1].copy()

    def feed_forward_generator(
        self,
        advantages,
        num_mini_batch=None,
        mini_batch_size=None,
        critic_obs_process_func=None,
    ):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * (episode_length - 1) * num_agents

        if mini_batch_size is None:
            assert (
                batch_size >= num_mini_batch
            ), (
                "DQN requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of DQN mini batches ({})."
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
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        if self._mixed_obs:
            critic_obs = {}
            policy_obs = {}
            next_critic_obs = {}
            next_policy_obs = {}
            for key in self.critic_obs.keys():
                critic_obs[key] = self.critic_obs[key][:-1].reshape(
                    -1, *self.critic_obs[key].shape[3:]
                )
            for key in self.policy_obs.keys():
                policy_obs[key] = self.policy_obs[key][:-1].reshape(
                    -1, *self.policy_obs[key].shape[3:]
                )
            for key in self.next_critic_obs.keys():
                next_critic_obs[key] = self.next_critic_obs[key][:-1].reshape(
                    -1, *self.next_critic_obs[key].shape[3:]
                )
            for key in self.next_policy_obs.keys():
                next_policy_obs[key] = self.next_policy_obs[key][:-1].reshape(
                    -1, *self.next_policy_obs[key].shape[3:]
                )
        else:
            critic_obs = self.critic_obs[:-1].reshape(-1, *self.critic_obs.shape[3:])
            policy_obs = self.policy_obs[:-1].reshape(-1, *self.policy_obs.shape[3:])
            next_critic_obs = self.next_critic_obs[:-1].reshape(
                -1, *self.next_critic_obs.shape[3:]
            )
            next_policy_obs = self.next_policy_obs[:-1].reshape(
                -1, *self.next_policy_obs.shape[3:]
            )

        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.action_masks is not None:
            action_masks = self.action_masks[:-1].reshape(
                -1, self.action_masks.shape[-1]
            )
        value_preds = self.value_preds[:-1].reshape(-1, self.act_space)
        rewards = self.rewards.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        next_masks = self.masks[1:].reshape(-1, 1)
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
                next_critic_obs_batch = {}
                next_policy_obs_batch = {}
                for key in critic_obs.keys():
                    critic_obs_batch[key] = critic_obs[key][indices]
                for key in policy_obs.keys():
                    policy_obs_batch[key] = policy_obs[key][indices]
                for key in next_critic_obs.keys():
                    next_critic_obs_batch[key] = next_critic_obs[key][indices]
                for key in next_policy_obs.keys():
                    next_policy_obs_batch[key] = next_policy_obs[key][indices]
            else:
                critic_obs_batch = critic_obs[indices]
                policy_obs_batch = policy_obs[indices]
                next_critic_obs_batch = next_critic_obs[indices]
                next_policy_obs_batch = next_policy_obs[indices]

            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.action_masks is not None:
                action_masks_batch = action_masks[indices]
            else:
                action_masks_batch = None
            value_preds_batch = value_preds[indices]
            rewards_batch = rewards[indices]
            masks_batch = masks[indices]
            next_masks_batch = next_masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = rewards_batch
            else:
                adv_targ = advantages[indices]
            if critic_obs_process_func is not None:
                critic_obs_batch = critic_obs_process_func(critic_obs_batch)

            yield critic_obs_batch, policy_obs_batch, next_critic_obs_batch, next_policy_obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, rewards_batch, masks_batch, next_masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, action_masks_batch
