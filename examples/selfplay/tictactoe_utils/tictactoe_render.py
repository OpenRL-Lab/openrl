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
import time
from typing import Optional, Union

import pygame
from pettingzoo.utils.env import ActionType, AECEnv, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper

from .game import Game


class TictactoeRender(BaseWrapper):
    def __init__(self, env: AECEnv):
        super().__init__(env)

        self.game = Game()
        self.last_action = None
        self.last_length = 0
        self.render_mode = "game"

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed, options)
        if self.render_mode == "game":
            self.game.reset()
            pygame.display.update()
            time.sleep(0.3)

        self.last_action = None

    def step(self, action: ActionType) -> None:
        result = super().step(action)
        self.last_action = action
        return result

    def observe(self, agent: str) -> Optional[ObsType]:
        obs = super().observe(agent)
        if self.last_action is not None:
            if self.render_mode == "game":
                self.game.make_move(self.last_action // 3, self.last_action % 3)
                pygame.display.update()
            self.last_action = None
            time.sleep(0.3)
        return obs

    def close(self):
        super().close()
        self.game.close()

    def set_render_mode(self, render_mode: Union[None, str]):
        self.render_mode = render_mode

    def get_human_action(self, agent, observation, termination, truncation, info):
        return self.game.get_human_action(
            agent, observation, termination, truncation, info
        )
