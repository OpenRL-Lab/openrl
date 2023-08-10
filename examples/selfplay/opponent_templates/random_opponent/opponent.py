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
from openrl.selfplay.opponents.random_opponent import RandomOpponent as Opponent

if __name__ == "__main__":
    from pettingzoo.classic import tictactoe_v3

    opponent1 = Opponent()
    opponent2 = Opponent()
    env = tictactoe_v3.env(render_mode="human")
    opponent1.reset(env, "player_1")
    opponent2.reset(env, "player_2")
    player2opponent = {"player_1": opponent1, "player_2": opponent2}

    env.reset()
    for player_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            break
        action = player2opponent[player_name].act(
            player_name, observation, reward, termination, truncation, info
        )
        print(player_name, action, type(action))
        env.step(action)
