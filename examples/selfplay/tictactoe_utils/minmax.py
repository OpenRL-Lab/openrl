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

import copy
import random
import sys

import numpy as np
import pygame
from constants import *
from game import (
    CIRC_COLOR,
    COLS,
    CROSS_COLOR,
    CROSS_WIDTH,
    HEIGHT,
    LINE_WIDTH,
    ROWS,
    SQSIZE,
    WIDTH,
    Game,
)

# --- CLASSES ---


class Board:
    def __init__(self):
        self.squares = np.zeros((ROWS, COLS))
        self.empty_sqrs = self.squares  # [squares]
        self.marked_sqrs = 0

    def final_state(self, show=False):
        """
        @return 0 if there is no win yet
        @return 1 if player 1 wins
        @return 2 if player 2 wins
        """

        # vertical wins
        for col in range(COLS):
            if (
                self.squares[0][col]
                == self.squares[1][col]
                == self.squares[2][col]
                != 0
            ):
                # if show:
                #     color = CIRC_COLOR if self.squares[0][col] == 2 else CROSS_COLOR
                #     iPos = (col * SQSIZE + SQSIZE // 2, 20)
                #     fPos = (col * SQSIZE + SQSIZE // 2, HEIGHT - 20)
                #     pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[0][col]

        # horizontal wins
        for row in range(ROWS):
            if (
                self.squares[row][0]
                == self.squares[row][1]
                == self.squares[row][2]
                != 0
            ):
                # if show:
                #     color = CIRC_COLOR if self.squares[row][0] == 2 else CROSS_COLOR
                #     iPos = (20, row * SQSIZE + SQSIZE // 2)
                #     fPos = (WIDTH - 20, row * SQSIZE + SQSIZE // 2)
                #     pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[row][0]

        # desc diagonal
        if self.squares[0][0] == self.squares[1][1] == self.squares[2][2] != 0:
            # if show:
            #     color = CIRC_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
            #     iPos = (20, 20)
            #     fPos = (WIDTH - 20, HEIGHT - 20)
            #     pygame.draw.line(screen, color, iPos, fPos, CROSS_WIDTH)
            return self.squares[1][1]

        # asc diagonal
        if self.squares[2][0] == self.squares[1][1] == self.squares[0][2] != 0:
            # if show:
            #     color = CIRC_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
            #     iPos = (20, HEIGHT - 20)
            #     fPos = (WIDTH - 20, 20)
            #     pygame.draw.line(screen, color, iPos, fPos, CROSS_WIDTH)
            return self.squares[1][1]

        # no win yet
        return 0

    def mark_sqr(self, row, col, player):
        self.squares[row][col] = player
        self.marked_sqrs += 1

    def empty_sqr(self, row, col):
        return self.squares[row][col] == 0

    def get_empty_sqrs(self):
        empty_sqrs = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.empty_sqr(row, col):
                    empty_sqrs.append((row, col))

        return empty_sqrs

    def isfull(self):
        return self.marked_sqrs == 9

    def isempty(self):
        return self.marked_sqrs == 0


class MINIMAXAlgorithm:
    def __init__(self, level=1, player=2):
        self.level = level
        self.player = player

    # --- RANDOM ---

    def rnd(self, board):
        empty_sqrs = board.get_empty_sqrs()
        idx = random.randrange(0, len(empty_sqrs))

        return empty_sqrs[idx]  # (row, col)

    # --- MINIMAX ---

    def minimax(self, board, maximizing):
        # terminal case
        case = board.final_state()

        # player 1 wins
        if case == 1:
            return 1, None  # eval, move

        # player 2 wins
        if case == 2:
            return -1, None

        # draw
        elif board.isfull():
            return 0, None

        if maximizing:
            max_eval = -100
            best_move = None
            empty_sqrs = board.get_empty_sqrs()

            for row, col in empty_sqrs:
                temp_board = copy.deepcopy(board)
                temp_board.mark_sqr(row, col, 1)
                eval = self.minimax(temp_board, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = (row, col)

            return max_eval, best_move

        elif not maximizing:
            min_eval = 100
            best_move = None
            empty_sqrs = board.get_empty_sqrs()

            for row, col in empty_sqrs:
                temp_board = copy.deepcopy(board)
                temp_board.mark_sqr(row, col, self.player)
                eval = self.minimax(temp_board, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = (row, col)

            return min_eval, best_move

    # --- MAIN EVAL ---

    def eval(self, main_board):
        if self.level == 0:
            # random choice
            eval = "random"
            move = self.rnd(main_board)
        else:
            # minimax algo choice
            eval, move = self.minimax(main_board, False)

        print(f"AI has chosen to mark the square in pos {move} with an eval of: {eval}")

        return move  # row, col


def main():
    # --- OBJECTS ---

    game = Game()
    board = game.board
    ai = MINIMAXAlgorithm()

    # --- MAINLOOP ---

    while True:
        # pygame events
        for event in pygame.event.get():
            # quit event
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # keydown event
            if event.type == pygame.KEYDOWN:
                # g-gamemode
                if event.key == pygame.K_g:
                    game.change_gamemode()

                # r-restart
                if event.key == pygame.K_r:
                    game.reset()
                    board = game.board
                    ai = game.ai

                # 0-random ai
                if event.key == pygame.K_0:
                    ai.level = 0

                # 1-random ai
                if event.key == pygame.K_1:
                    ai.level = 1

            # click event
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                row = pos[1] // SQSIZE
                col = pos[0] // SQSIZE

                # human mark sqr
                if board.empty_sqr(row, col) and game.running:
                    game.make_move(row, col)

                    if game.isover():
                        game.running = False

        # AI initial call
        if game.gamemode == "ai" and game.player == ai.player and game.running:
            # update the screen
            pygame.display.update()

            # eval
            row, col = ai.eval(board)
            game.make_move(row, col)

            if game.isover():
                game.running = False

        pygame.display.update()
