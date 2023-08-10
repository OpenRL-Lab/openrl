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
from pathlib import Path
from typing import Dict, Union

import numpy as np
from gymnasium import spaces

from openrl.envs.wrappers.flatten import flatten
from openrl.modules.common.ppo_net import PPONet
from openrl.runners.common.ppo_agent import PPOAgent
from openrl.selfplay.opponents.network_opponent import NetworkOpponent
from openrl.selfplay.opponents.opponent_env import BaseOpponentEnv


class TicTacToeOpponentEnv(BaseOpponentEnv):
    def __init__(self, env, opponent_player: str):
        super().__init__(env, opponent_player)
        self.middle_observation_space = self.env.observation_space(
            self.opponent_player
        ).spaces["observation"]
        self.observation_space = spaces.flatten_space(self.middle_observation_space)

    def process_obs(self, observation, termination, truncation, info):
        new_obs = observation["observation"]
        new_info = info.copy()
        new_info["action_masks"] = observation["action_mask"][np.newaxis, ...]
        new_obs = flatten(self.middle_observation_space, self.agent_num, new_obs)
        new_obs = new_obs[np.newaxis, ...]
        new_info = [new_info]

        return new_obs, termination, truncation, new_info

    def process_action(self, action):
        return action[0][0][0]


class Opponent(NetworkOpponent):
    def __init__(
        self,
        opponent_id: str,
        opponent_path: Union[str, Path],
        opponent_info: Dict[str, str],
    ):
        super().__init__(opponent_id, opponent_path, opponent_info)
        self.deterministic_action = False

    def _set_env(self, env, opponent_player: str):
        self.opponent_env = TicTacToeOpponentEnv(env, opponent_player)
        self.agent = PPOAgent(PPONet(self.opponent_env))
        self.load(self.opponent_path)

    def _load(self, opponent_path: Union[str, Path]):
        model_path = Path(opponent_path) / "module.pt"
        if self.agent is not None:
            self.agent.load(model_path)


def test_opponent():
    from pettingzoo.classic import tictactoe_v3

    opponent = Opponent(
        "1", "./", opponent_info={"opponent_type": "tictactoe_opponent"}
    )
    env = tictactoe_v3.env()
    opponent.load("./")
    opponent.reset(env, "player_1")

    env.reset()
    for player_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            break
        action = opponent.act(
            player_name, observation, reward, termination, truncation, info
        )
        print(player_name, action, type(action))
        env.step(action)


if __name__ == "__main__":
    test_opponent()
