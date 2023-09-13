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

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from concurrent.futures import as_completed
from typing import Any, Callable, Dict, Optional

from gymnasium.vector.utils import CloudpickleWrapper
from tqdm.rich import tqdm

from openrl.arena.agents.base_agent import BaseAgent
from openrl.arena.games.base_game import BaseGame


class BaseArena(ABC):
    def __init__(
        self,
        env_fn: Callable,
        dispatch_func: Optional[Callable] = None,
        use_tqdm: bool = True,
    ):
        self.env_fn = env_fn
        self.pbar = None

        self.dispatch_func = dispatch_func

        self.total_games = None
        self.max_game_onetime = None
        self.agents = None
        self.game: Optional[BaseGame] = None
        self.seed = None
        self.use_tqdm = use_tqdm

    def reset(
        self,
        agents: Dict[str, BaseAgent],
        total_games: int,
        max_game_onetime: int = 5,
        seed: int = 0,
        dispatch_func: Optional[Callable] = None,
    ):
        self.seed = seed
        if self.pbar:
            self.pbar.refresh()
            self.pbar.close()
        if self.use_tqdm:
            self.pbar = tqdm(total=total_games, desc="Processing")
        self.total_games = total_games
        self.max_game_onetime = max_game_onetime
        self.agents = agents
        assert isinstance(self.game, BaseGame)

        if dispatch_func is not None:
            self.dispatch_func = dispatch_func

        self.game.reset(seed=seed, dispatch_func=self.dispatch_func)

    def close(self):
        if self.pbar:
            self.pbar.refresh()
            self.pbar.close()

    def _run_parallel(self):
        with PoolExecutor(
            max_workers=min(self.max_game_onetime, self.total_games)
        ) as executor:
            futures = [
                executor.submit(
                    self.game.run,
                    self.seed + run_index,
                    CloudpickleWrapper(self.env_fn),
                    self.agents,
                )
                for run_index in range(self.total_games)
            ]
            for future in as_completed(futures):
                result = future.result()
                self._deal_result(result)
                if self.pbar:
                    self.pbar.update(1)

    def _run_serial(self):
        for run_index in range(self.total_games):
            result = self.game.run(self.seed + run_index, self.env_fn, self.agents)
            self._deal_result(result)
            if self.pbar:
                self.pbar.update(1)

    def run(self, parallel: bool = True) -> Dict[str, Any]:
        assert self.seed is not None, "Please call reset() to set seed first."
        if parallel:
            self._run_parallel()
        else:
            self._run_serial()
        return self._get_final_result()

    @abstractmethod
    def _deal_result(self, result: Any):
        pass

    @abstractmethod
    def _get_final_result(self) -> Dict[str, Any]:
        raise NotImplementedError
