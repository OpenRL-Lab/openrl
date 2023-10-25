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
import os
import sys

import pytest

from openrl.arena import make_arena
from openrl.arena.agents.local_agent import LocalAgent
from openrl.arena.agents.random_agent import RandomAgent
from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner


def run_arena(
    render: bool = False,
    parallel: bool = True,
    seed=0,
    total_games: int = 10,
    max_game_onetime: int = 5,
):
    env_wrappers = [RecordWinner]
    if render:
        from examples.selfplay.tictactoe_utils.tictactoe_render import TictactoeRender

        env_wrappers.append(TictactoeRender)

    arena = make_arena("tictactoe_v3", env_wrappers=env_wrappers, use_tqdm=False)

    agent1 = LocalAgent("./examples/selfplay/opponent_templates/random_opponent")
    agent2 = RandomAgent()

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


@pytest.mark.unittest
def test_seed():
    seed = 0
    test_time = 3
    pre_result = None
    for parallel in [False, True]:
        for i in range(test_time):
            result = run_arena(seed=seed, parallel=parallel, total_games=20)
            if pre_result is not None:
                assert pre_result == result, f"parallel={parallel}, seed={seed}"
            pre_result = result


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
