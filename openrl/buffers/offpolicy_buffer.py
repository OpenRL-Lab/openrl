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

from .normal_buffer import NormalReplayBuffer
from .offpolicy_replay_data import OffPolicyReplayData


class OffPolicyReplayBuffer(NormalReplayBuffer):
    def __init__(
        self, cfg, num_agents, obs_space, act_space, data_client, episode_length=None
    ):
        if episode_length is None:
            episode_length = cfg.episode_length
        self.buffer_size = cfg.episode_length
        self.data = OffPolicyReplayData(
            cfg,
            num_agents,
            obs_space,
            act_space,
            data_client,
            episode_length,
        )

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
        self.data.insert(
            raw_obs,
            # next_raw_obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks,
            active_masks,
            action_masks,
        )

    def get_buffer_size(self):
        if self.data.first_insert_flag:
            return self.data.step
        else:
            return self.buffer_size
