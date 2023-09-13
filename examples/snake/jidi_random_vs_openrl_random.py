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

from openrl.arena import make_arena
from openrl.arena.agents.jidi_agent import JiDiAgent
from openrl.arena.agents.local_agent import LocalAgent
from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner


def run_arena(
    render: bool = False,
    parallel: bool = True,
    seed=0,
    total_games: int = 10,
    max_game_onetime: int = 5,
):
    env_wrappers = [RecordWinner]

    player_num = 3
    arena = make_arena(
        f"snakes_{player_num}v{player_num}",
        env_wrappers=env_wrappers,
        render=render,
        use_tqdm=True,
    )

    agent1 = JiDiAgent("./submissions/random_agent", player_num=player_num)
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


if __name__ == "__main__":
    run_arena(render=False, parallel=True, seed=0, total_games=100, max_game_onetime=5)
