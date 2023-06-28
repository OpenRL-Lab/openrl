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
from typing import Any, Dict, Tuple

from openrl.drivers.onpolicy_driver import OnPolicyDriver


class GAILDriver(OnPolicyDriver):
    def actor_rollout(self) -> Tuple[Dict[str, Any], bool]:
        self.callback.on_rollout_start()

        self.trainer.prep_rollout()

        for step in range(self.episode_length):
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.act(
                step
            )

            extra_data = {
                "actions": actions,
                "values": values,
                "action_log_probs": action_log_probs,
                "step": step,
                "buffer": self.buffer,
            }

            obs, rewards, dones, infos = self.envs.step(actions, extra_data)
            self.agent.num_time_steps += self.envs.parallel_env_num
            # Give access to local variables
            self.callback.update_locals(locals())
            if self.callback.on_step() is False:
                return {}, False

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

        self.callback.on_rollout_end()

        if self.envs.use_monitor:
            statistics_info = self.envs.statistics(self.buffer)
            statistics_info.update(batch_rew_infos)
            return statistics_info, True
        else:
            return batch_rew_infos, True
