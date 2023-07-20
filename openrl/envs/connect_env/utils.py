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


def check_if_win(state, check_row_pos, check_col_pos, env):
    def check_if_win_direction(now_state, direction, row_pos, col_pos, env):
        def check_if_valid(x_pos, y_pos):
            return (
                x_pos >= 0
                and x_pos <= (env.row - 1)
                and y_pos >= 0
                and y_pos <= (env.col - 1)
            )

        check_who = now_state[row_pos][col_pos]
        counting = 1
        bias_num = 1
        while True:
            new_row_pos = row_pos + bias_num * direction[0]
            new_col_pos = col_pos + bias_num * direction[1]
            if (
                not check_if_valid(new_row_pos, new_col_pos)
                or now_state[new_row_pos][new_col_pos] != check_who
            ):
                break
            else:
                counting += 1
                bias_num += 1
        bias_num = -1
        while True:
            new_row_pos = row_pos + bias_num * direction[0]
            new_col_pos = col_pos + bias_num * direction[1]
            if (
                not check_if_valid(new_row_pos, new_col_pos)
                or now_state[new_row_pos][new_col_pos] != check_who
            ):
                break
            else:
                counting += 1
                bias_num -= 1
        if counting >= env.num_to_win:
            return True
        else:
            return False

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横 竖 右下 右上
    for direction in directions:
        if check_if_win_direction(state, direction, check_row_pos, check_col_pos, env):
            return True
    return False
