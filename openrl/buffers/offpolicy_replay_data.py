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

from openrl.buffers.replay_data import ReplayData
from openrl.buffers.utils.obs_data import ObsData
from openrl.buffers.utils.util import get_critic_obs, get_policy_obs


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
        self.first_insert_flag = True

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
        if "available_actions" in data:
            self.available_actions[self.step + 1] = data["available_actions"].copy()

        if (self.step + 1) % self.episode_length != 0:
            self.first_insert_flag = False
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
        available_actions=None,
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

        self.rnn_states[self.step + 1] = rnn_states.copy()
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
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        if (self.step + 1) % self.episode_length != 0:
            self.first_insert_flag = False
        self.step = (self.step + 1) % self.episode_length

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
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        pass
