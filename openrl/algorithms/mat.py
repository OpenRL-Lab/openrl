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
from openrl.algorithms.ppo import PPOAlgorithm


class MATAlgorithm(PPOAlgorithm):
    def construct_loss_list(self, policy_loss, dist_entropy, value_loss, turn_on):
        loss_list = []

        loss = (
            policy_loss
            - dist_entropy * self.entropy_coef
            + value_loss * self.value_loss_coef
        )
        loss_list.append(loss)

        return loss_list

    def get_data_generator(self, buffer, advantages):
        data_generator = buffer.feed_forward_generator_transformer(
            advantages, self.num_mini_batch
        )
        return data_generator
