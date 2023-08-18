import os
import sys

import numpy as np


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Board:
    def __init__(self, board_height, board_width, snakes, beans_positions, teams):
        # print('create board, beans_position: ', beans_positions)
        self.height = board_height
        self.width = board_width
        self.snakes = snakes
        self.snakes_count = len(snakes)
        self.beans_positions = beans_positions
        self.blank_sign = -self.snakes_count
        self.bean_sign = -self.snakes_count + 1
        self.board = np.zeros((board_height, board_width), dtype=int) + self.blank_sign
        self.open = dict()
        for key, snake in self.snakes.items():
            self.open[key] = [snake.head]  # state 0 open list, heads, ready to spread
            # see [A* Pathfinding (E01: algorithm explanation)](https://www.youtube.com/watch?v=-L-WgKMFuhE)
            for x, y in snake.pos:
                self.board[x][y] = key  # obstacles, e.g. 0, 1, 2, 3, 4, 5
        # for x, y in beans_positions:
        #     self.board[x][y] = self.bean_sign  # beans

        self.state = 0
        self.controversy = dict()
        self.teams = teams

        # print('initial board')
        # print(self.board)

    def step(self):  # delay: prevent rear-end collision
        new_open = {key: [] for key in self.snakes.keys()}
        self.state += 1  # update state
        # if self.state > delay:
        #     for key, snake in self.snakes.items():   # drop tail
        #         if snake.len >= self.state:
        #             self.board[snake.pos[-(self.state - delay)][0]][snake.pos[-(self.state - delay)][1]] \
        #                 = self.blank_sign
        for key, snake in self.snakes.items():
            if snake.len >= self.state:
                self.board[snake.pos[-self.state][0]][
                    snake.pos[-self.state][1]
                ] = self.blank_sign  # drop tail
        for key, value in self.open.items():  # value: e.g. [[8, 3], [6, 3], [7, 4]]
            others_tail_pos = [
                (
                    self.snakes[_].pos[-self.state]
                    if self.snakes[_].len >= self.state
                    else []
                )
                for _ in set(range(self.snakes_count)) - {key}
            ]
            for x, y in value:
                # print('start to spread snake {} on grid ({}, {})'.format(key, x, y))
                for x_, y_ in [
                    ((x + 1) % self.height, y),  # down
                    ((x - 1) % self.height, y),  # up
                    (x, (y + 1) % self.width),  # right
                    (x, (y - 1) % self.width),
                ]:  # left
                    sign = self.board[x_][y_]
                    idx = (
                        sign % self.snakes_count
                    )  # which snake, e.g. 0, 1, 2, 3, 4, 5 / number of claims
                    state = (
                        sign // self.snakes_count
                    )  # manhattan distance to snake who claim the point or its negative
                    if sign == self.blank_sign:  # grid in initial state
                        if [x_, y_] in others_tail_pos:
                            # print('do not spread other snakes tail, in case of rear-end collision')
                            continue  # do not spread other snakes' tail, in case of rear-end collision
                        self.board[x_][y_] = self.state * self.snakes_count + key
                        self.snakes[key].claimed_count += 1
                        new_open[key].append([x_, y_])

                    elif key != idx and self.state == state:
                        # second claim, init controversy, change grid value from + to -
                        # print(
                        #     '\tgird ({}, {}) in the same state claimed by different snakes '
                        #     'with sign {}, idx {} and state {}'.format(
                        #         x_, y_, sign, idx, state))
                        if (
                            self.snakes[idx].len > self.snakes[key].len
                        ):  # shorter snake claim the controversial grid
                            # print('\t\tsnake {} is shorter than snake {}'.format(key, idx))
                            self.snakes[idx].claimed_count -= 1
                            new_open[idx].remove([x_, y_])
                            self.board[x_][y_] = self.state * self.snakes_count + key
                            self.snakes[key].claimed_count += 1
                            new_open[key].append([x_, y_])
                        elif (
                            self.snakes[idx].len == self.snakes[key].len
                        ):  # controversial claim
                            # print(
                            #     '\t\tcontroversy! first claimed by snake {}, then claimed by snake {}'.format(idx, key))
                            self.controversy[(x_, y_)] = {
                                "state": self.state,
                                "length": self.snakes[idx].len,
                                "indexes": [idx, key],
                            }
                            # first claim by snake idx, then claim by snake key
                            self.board[x_][y_] = -self.state * self.snakes_count + 1
                            # if + 2, not enough for all snakes claim one grid!!
                            self.snakes[
                                idx
                            ].claimed_count -= (
                                1  # controversy, no snake claim this grid!!
                            )
                            new_open[key].append([x_, y_])
                        else:  # (self.snakes[idx].len < self.snakes[key].len)
                            pass  # longer snake do not claim the controversial grid

                    elif (
                        (x_, y_) in self.controversy
                        and key not in self.controversy[(x_, y_)]["indexes"]
                        and self.state + state == 0
                    ):  # third claim or more
                        # print('snake {} meets third or more claim in grid ({}, {})'.format(key, x_, y_))
                        controversy = self.controversy[(x_, y_)]
                        # pprint.pprint(controversy)
                        if (
                            controversy["length"] > self.snakes[key].len
                        ):  # shortest snake claim grid, do 4 things
                            # print('\t\tsnake {} is shortest'.format(key))
                            indexes_count = len(controversy["indexes"])
                            for i in controversy["indexes"]:
                                self.snakes[i].claimed_count -= (
                                    1 / indexes_count
                                )  # update claimed_count !
                                new_open[i].remove([x_, y_])
                            del self.controversy[(x_, y_)]
                            self.board[x_][y_] = self.state * self.snakes_count + key
                            self.snakes[key].claimed_count += 1
                            new_open[key].append([x_, y_])
                        elif (
                            controversy["length"] == self.snakes[key].len
                        ):  # controversial claim
                            # print('\t\tcontroversy! multi claimed by snake {}'.format(key))
                            self.controversy[(x_, y_)]["indexes"].append(key)
                            self.board[x_][y_] += 1
                            new_open[key].append([x_, y_])
                        else:  # (controversy['length'] < self.snakes[key].len)
                            pass  # longer snake do not claim the controversial grid
                    else:
                        pass  # do nothing with lower state grids

        self.open = new_open  # update open
        # update controversial snakes' claimed_count (in fraction) in the end
        for _, d in self.controversy.items():
            controversial_snake_count = len(
                d["indexes"]
            )  # number of controversial snakes
            for idx in d["indexes"]:
                self.snakes[idx].claimed_count += 1 / controversial_snake_count


class SnakePos:
    def __init__(self, snake_positions, board_height, board_width, beans_positions):
        self.pos = snake_positions  # [[2, 9], [2, 8], [2, 7]]
        self.len = len(snake_positions)  # >= 3
        self.head = snake_positions[0]
        self.beans_positions = beans_positions
        self.claimed_count = 0

        displace = [
            (self.head[0] - snake_positions[1][0]) % board_height,
            (self.head[1] - snake_positions[1][1]) % board_width,
        ]
        # print('creat snake, pos: ', self.pos, 'displace:', displace)
        if displace == [
            board_height - 1,
            0,
        ]:  # all action are ordered by left, up, right, relative to the body
            self.dir = 0  # up
            self.legal_action = [2, 0, 3]
        elif displace == [1, 0]:
            self.dir = 1  # down
            self.legal_action = [3, 1, 2]
        elif displace == [0, board_width - 1]:
            self.dir = 2  # left
            self.legal_action = [1, 2, 0]
        elif displace == [0, 1]:
            self.dir = 3  # right
            self.legal_action = [0, 3, 1]
        else:
            assert False, "snake positions error"
        positions = [
            [(self.head[0] - 1) % board_height, self.head[1]],
            [(self.head[0] + 1) % board_height, self.head[1]],
            [self.head[0], (self.head[1] - 1) % board_width],
            [self.head[0], (self.head[1] + 1) % board_width],
        ]
        self.legal_position = [positions[_] for _ in self.legal_action]

    def get_action(self, position):
        if position not in self.legal_position:
            assert False, "the start and end points do not match"
        idx = self.legal_position.index(position)
        return self.legal_action[idx]  # 0, 1, 2, 3: up, down, left, right

    def step(self, legal_input):
        if legal_input in self.legal_position:
            position = legal_input
        elif legal_input in self.legal_action:
            idx = self.legal_action.index(legal_input)
            position = self.legal_position[idx]
        else:
            assert False, "illegal snake move"
        self.head = position
        self.pos.insert(0, position)
        if position in self.beans_positions:  # eat a bean
            self.len += 1
        else:  # do not eat a bean
            self.pos.pop()
