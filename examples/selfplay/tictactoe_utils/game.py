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

import sys

import pygame

WIDTH = 600
HEIGHT = 600

ROWS = 3
COLS = 3
SQSIZE = WIDTH // COLS

LINE_WIDTH = 15
CIRC_WIDTH = 15
CROSS_WIDTH = 20

RADIUS = SQSIZE // 4

OFFSET = 50

# --- COLORS ---

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRC_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)


class Game:
    def __init__(self):
        self.screen = None

    def reset(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("TIC TAC TOE")
            self.screen.fill(BG_COLOR)

        self.player = 1  # 1-cross  #2-circles
        self.running = True
        self.show_lines()

    # --- DRAW METHODS ---
    def show_lines(self):
        # bg
        self.screen.fill(BG_COLOR)

        # vertical
        pygame.draw.line(
            self.screen, LINE_COLOR, (SQSIZE, 0), (SQSIZE, HEIGHT), LINE_WIDTH
        )
        pygame.draw.line(
            self.screen,
            LINE_COLOR,
            (WIDTH - SQSIZE, 0),
            (WIDTH - SQSIZE, HEIGHT),
            LINE_WIDTH,
        )

        # horizontal
        pygame.draw.line(
            self.screen, LINE_COLOR, (0, SQSIZE), (WIDTH, SQSIZE), LINE_WIDTH
        )
        pygame.draw.line(
            self.screen,
            LINE_COLOR,
            (0, HEIGHT - SQSIZE),
            (WIDTH, HEIGHT - SQSIZE),
            LINE_WIDTH,
        )

    def draw_fig(self, row, col):
        if self.player == 1:
            # draw cross
            # desc line
            start_desc = (col * SQSIZE + OFFSET, row * SQSIZE + OFFSET)
            end_desc = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            pygame.draw.line(
                self.screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH
            )
            # asc line
            start_asc = (col * SQSIZE + OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            end_asc = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + OFFSET)
            pygame.draw.line(self.screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)

        elif self.player == 2:
            # draw circle
            center = (col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2)
            pygame.draw.circle(self.screen, CIRC_COLOR, center, RADIUS, CIRC_WIDTH)

    # --- OTHER METHODS ---

    def make_move(self, row, col):
        self.draw_fig(row, col)
        self.next_turn()

    def next_turn(self):
        self.player = self.player % 2 + 1

    def close(self):
        try:
            self.screen.fill((0, 0, 0, 0))
            pygame.display.update()
            del self.screen
            pygame.quit()
        except Exception:
            pass

    def get_human_action(self, agent, observation, termination, truncation, info):
        action_mask = observation["action_mask"]
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    row = pos[1] // SQSIZE
                    col = pos[0] // SQSIZE
                    action = row * 3 + col
                    if action_mask[action]:
                        return action
