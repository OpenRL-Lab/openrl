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

from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.classic import connect_four_v3, go_v5, texas_holdem_no_limit_v6,rps_v2
from pettingzoo.mpe import simple_push_v3


from openrl.arena import make_arena
from openrl.arena.agents.local_agent import LocalAgent
from openrl.envs.PettingZoo.registration import register
from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner


def ConnectFourEnv(render_mode, **kwargs):
    return connect_four_v3.env(render_mode)


def RockPaperScissorsEnv(render_mode, **kwargs):
    return rps_v2.env(num_actions=3, max_cycles=15)


def GoEnv(render_mode, **kwargs):
    return go_v5.env(render_mode=render_mode, board_size=5, komi=7.5)


def TexasHoldemEnv(render_mode, **kwargs):
    return texas_holdem_no_limit_v6.env(render_mode=render_mode)


# MPE
def SimplePushEnv(render_mode, **kwargs):
    return simple_push_v3.env(render_mode=render_mode)


def CooperativePongEnv(render_mode, **kwargs):
    return cooperative_pong_v5.env(render_mode=render_mode)


def register_new_envs():
    new_env_dict = {
        "connect_four_v3": ConnectFourEnv,
        "RockPaperScissors": RockPaperScissorsEnv,
        "go_v5": GoEnv,
        "texas_holdem_no_limit_v6": TexasHoldemEnv,
        "simple_push_v3": SimplePushEnv,
        "cooperative_pong_v5": CooperativePongEnv,
    }

    for env_id, env in new_env_dict.items():
        register(env_id, env)
    return new_env_dict.keys()


def run_arena(
    env_id: str,
    parallel: bool = True,
    seed=0,
    total_games: int = 10,
    max_game_onetime: int = 5,
):
    env_wrappers = [RecordWinner]

    arena = make_arena(env_id, env_wrappers=env_wrappers, use_tqdm=False)

    agent1 = LocalAgent("../selfplay/opponent_templates/random_opponent")
    agent2 = LocalAgent("../selfplay/opponent_templates/random_opponent")

    arena.reset(
        agents={"agent1": agent1, "agent2": agent2},
        total_games=total_games,
        max_game_onetime=max_game_onetime,
        seed=seed,
    )
    result = arena.run(parallel=parallel)
    arena.close()
    print(result)
    return result


def test_new_envs():
    env_ids = register_new_envs()
    seed = 0
    for env_id in env_ids:
        run_arena(env_id=env_id, seed=seed, parallel=False, total_games=1)


if __name__ == "__main__":
    test_new_envs()
