# code from https://github.com/CarlossShi/Competition_3v3snakes/blob/master/agent/submit/submission.py
import copy
import itertools
import operator
import os
import pprint
import random
import sys

import numpy as np


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_direction(x_h, y_h, x, y, height, width):  # from (x_h, y_h) to (x, y)
    if (x_h + 1) % height == x and y_h == y:
        return [0, 1, 0, 0]
    elif (x_h - 1) % height == x and y_h == y:
        return [1, 0, 0, 0]
    elif x_h == x and (y_h + 1) % width == y:
        return [0, 0, 0, 1]
    elif x_h == x and (y_h - 1) % width == y:
        return [0, 0, 1, 0]
    else:
        assert False, "the start and end points do not match"


def connected_count(matrix, pos):
    height, width = matrix.shape
    x, y = pos
    sign = matrix[x, y]
    unexplored = [[x, y]]
    explored = []
    _connected_count = 1
    while unexplored:
        x, y = unexplored.pop()
        explored.append([x, y])
        for x_, y_ in [
            ((x + 1) % height, y),  # down
            ((x - 1) % height, y),  # up
            (x, (y + 1) % width),  # right
            (x, (y - 1) % width),
        ]:  # left
            if (
                matrix[x_, y_] == sign
                and [x_, y_] not in explored
                and [x_, y_] not in unexplored
            ):
                unexplored.append([x_, y_])
                _connected_count += 1
    return _connected_count


class Snake:
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
        print("creat snake, pos: ", self.pos, "displace:", displace)
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


class Board:
    def __init__(self, board_height, board_width, snakes, beans_positions, teams):
        print("create board, beans_position: ", beans_positions)
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

        print("initial board")
        print(self.board)

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
                print("start to spread snake {} on grid ({}, {})".format(key, x, y))
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
                            print(
                                "do not spread other snakes tail, in case of rear-end"
                                " collision"
                            )
                            continue  # do not spread other snakes' tail, in case of rear-end collision
                        self.board[x_][y_] = self.state * self.snakes_count + key
                        self.snakes[key].claimed_count += 1
                        new_open[key].append([x_, y_])

                    elif key != idx and self.state == state:
                        # second claim, init controversy, change grid value from + to -
                        print(
                            "\tgird ({}, {}) in the same state claimed by different"
                            " snakes with sign {}, idx {} and state {}".format(
                                x_, y_, sign, idx, state
                            )
                        )
                        if (
                            self.snakes[idx].len > self.snakes[key].len
                        ):  # shorter snake claim the controversial grid
                            print(
                                "\t\tsnake {} is shorter than snake {}".format(key, idx)
                            )
                            self.snakes[idx].claimed_count -= 1
                            new_open[idx].remove([x_, y_])
                            self.board[x_][y_] = self.state * self.snakes_count + key
                            self.snakes[key].claimed_count += 1
                            new_open[key].append([x_, y_])
                        elif (
                            self.snakes[idx].len == self.snakes[key].len
                        ):  # controversial claim
                            print(
                                "\t\tcontroversy! first claimed by snake {}, then"
                                " claimed by snake {}".format(idx, key)
                            )
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
                        print(
                            "snake {} meets third or more claim in grid ({}, {})"
                            .format(key, x_, y_)
                        )
                        controversy = self.controversy[(x_, y_)]
                        pprint.pprint(controversy)
                        if (
                            controversy["length"] > self.snakes[key].len
                        ):  # shortest snake claim grid, do 4 things
                            print("\t\tsnake {} is shortest".format(key))
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
                            print(
                                "\t\tcontroversy! multi claimed by snake {}".format(key)
                            )
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

    def claim2action(self, claim_position, snake_idx, step_count, output_type):
        # claim e.g. [2 ,3 ,4 ,-9] bean 2 is claimed by snake 3 within 4 steps
        x, y = claim_position
        x_h, y_h = self.snakes[snake_idx].head  # head position

        while step_count > 1:
            step_count -= 1
            temp = []
            for x_, y_ in [
                ((x + 1) % self.height, y),  # down
                ((x - 1) % self.height, y),  # up
                (x, (y + 1) % self.width),  # right
                (x, (y - 1) % self.width),
            ]:  # left
                sign = self.board[x_][y_]
                if sign == self.blank_sign:
                    continue  # snake too long, board not spread completely!! see example 20210815 0:41:48
                state = (
                    sign // self.snakes_count
                    if sign > 0
                    else -(sign // self.snakes_count)
                )
                indexes = (
                    [sign % self.snakes_count]
                    if sign >= 0
                    else self.controversy[(x_, y_)]["indexes"]
                )
                if step_count == state and snake_idx in indexes:
                    temp.append([x_, y_])
            x, y = random.choice(temp)
        if output_type == "action":
            return get_direction(x_h, y_h, x, y, self.height, self.width)
        elif output_type == "position":
            return [x, y]
        else:
            assert False, "unknown output_type {}".format(output_type)


def state2claims(state_array, max_state, priority):
    # state_array:
    # array([[ 1, 30, 30, 30, 30, 30],
    #        [30, 30,  3, 30, 30, 30],
    #        [30,  1, 30,  5, 30, 30],
    #        [30, 30, 30,  6,  5, 30],
    #        [30, 30, 30, 30, 30,  6]])
    beanCount, snakeCount = state_array.shape  # (5, 6)
    horiz = [
        list(state_array[_]).count(max_state) for _ in range(beanCount)
    ]  # [5, 5, 4, 4, 5]
    vert = [
        list(state_array[:, _]).count(max_state) for _ in range(snakeCount)
    ]  # [4, 4, 4, 3, 4, 4]
    claim_order = []
    for b in range(beanCount):
        for s in range(snakeCount):
            if state_array[b][s] < max_state:
                claim_order.append([b, s, state_array[b][s], -horiz[b] - vert[s]])
    # priority rule: smaller state, larger horizontal or vertical cover number of max_state
    if not claim_order:
        return []
    if priority == "state":
        temp = min(claim_order, key=operator.itemgetter(2, 3))  # [0, 0, 1, -9]
    elif priority == "cover":
        temp = min(claim_order, key=operator.itemgetter(3, 2))
    else:
        assert False, "unknown priority"
    # update
    for b in range(beanCount):
        state_array[b, temp[1]] = max_state + 1
    for s in range(snakeCount):
        state_array[temp[0], s] = max_state + 1
    return [temp] + state2claims(state_array, max_state, priority)


def my_controller(observation_list, action_space_list, is_act_continuous):
    with HiddenPrints():
        # detect 1v1, 3v3, 2p or 5p
        # if True:
        observation_len = len(observation_list.keys())
        teams = None
        if observation_len == 7:
            teams = [[0], [1]]  # 1v1
            # teams = [[0, 1]]  # 2p
        elif observation_len == 10:
            teams = [[0, 1, 2, 3, 4]]  # 5p
        elif observation_len == 11:
            teams = [[0, 1, 2], [3, 4, 5]]  # 3v3

        assert teams is not None, "unknown game with observation length {}".format(
            observation_len
        )
        teams_count = len(teams)
        snakes_count = sum([len(_) for _ in teams])

        # read observation
        obs = observation_list.copy()
        board_height = obs["board_height"]  # 10
        board_width = obs["board_width"]  # 20
        ctrl_agent_index = obs["controlled_snake_index"] - 2  # 0, 1, 2, 3, 4, 5
        # last_directions = obs['last_direction']  # ['up', 'left', 'down', 'left', 'left', 'left']
        beans_positions = obs[1]  # e.g.[[7, 15], [4, 14], [5, 12], [4, 12], [5, 7]]
        snakes = {
            key - 2: Snake(obs[key], board_height, board_width, beans_positions)
            for key in obs.keys() & {_ + 2 for _ in range(snakes_count)}
        }  # &: intersection
        team_indexes = [_ for _ in teams if ctrl_agent_index in _][0]

        init_board = Board(board_height, board_width, snakes, beans_positions, teams)
        bd = copy.deepcopy(init_board)

        with HiddenPrints():
            while not all(
                _ == [] for _ in bd.open.values()
            ):  # loop until all values in open are empty list
                bd.step()
        print(bd.board)

        defense_snakes_indexes = (
            []
        )  # save defensive or claimed snakes, to calculate safe move for ctrl snake

        # define defensive move
        defensive_claim_list = []  # [pos, snake_idx, step]
        # first check win side
        snakes_lens = [snake.len for snake in snakes.values()]
        snakes_claimed_counts = [snake.len for snake in snakes.values()]
        print("snakes_lens: ", snakes_lens)
        print("snakes_claimed_counts: ", snakes_claimed_counts)

        # design defense threshold
        # defense_threshold = 0.5 * math.pow(board_height * board_width, 1.1) / snakes_count * \
        #                     math.sqrt(4 / (4 * teams_count + 1))
        defense_threshold = (
            board_height * board_width * teams_count / (teams_count + 1) / snakes_count
        )

        for idx in team_indexes:
            if snakes_lens[idx] > defense_threshold:
                # 3: player count + 1, 2: player count, 6: snake count
                for _ in range(1, min(bd.state, snakes_lens[idx] // 2)):
                    # range should be designed more carefully!!
                    x, y = snakes[idx].pos[-_]
                    if (
                        bd.board[x, y] == idx + _ * snakes_count
                    ):  # claim a loop in step _
                        defense_snakes_indexes.append(idx)
                        defensive_claim_list.append([[x, y], idx, _])
                        if idx == ctrl_agent_index:
                            action = [
                                bd.claim2action(
                                    claim_position=[x, y],
                                    snake_idx=idx,
                                    step_count=_,
                                    output_type="action",
                                )
                            ]
                            print(
                                "the controlled agent {} make a defensive move {}"
                                " within {} step(s)".format(idx, action[0], _)
                            )
                            print(
                                "***********************************"
                                + " defensive move "
                                + "***************************************"
                            )
                            return action

        # calculate state_array
        # e.g.
        # array([[ 1, 30, 30, 30, 30, 30],
        #        [30, 30,  3, 30, 30, 30],
        #        [30,  1, 30,  5, 30, 30],
        #        [30, 30, 30,  6,  5, 30],
        #        [30, 30, 30, 30, 30,  6]])
        max_state = board_height + board_width  # 30
        state_array = (
            np.zeros((len(beans_positions), len(snakes)), dtype=int) + max_state
        )
        for i, (x, y) in enumerate(beans_positions):
            sign = bd.board[x][y]
            if sign >= snakes_count:  # bean claimed by one snake
                idx = sign % snakes_count  # 0, 1, 2, 3, 4, 5
                state = sign // snakes_count  # 1, 2, ...
                state_array[i][idx] = state
            elif sign < 0 and sign % snakes_count in [
                _ for _ in range(snakes_count) if _ > 0
            ]:  # [2, 3, 4, 5]
                state = -(sign // snakes_count)
                for idx in bd.controversy[(x, y)]["indexes"]:
                    state_array[i][idx] = state
            elif (
                sign == bd.blank_sign
            ):  # bean not reachable for any snakes! see example: 20210815, 1:10:23
                pass
            else:
                assert False, "unknown sign when calculating state_array"

        # calculate claim list
        # e.g. [[2, 3, 4, -9], [1, 1, 4, -6], [3, 2, 4, -4], [0, 4, 5, -2]]
        claim_list_byState = state2claims(state_array.copy(), max_state, "state")
        claim_list_byCover = state2claims(state_array.copy(), max_state, "cover")
        print("claim_list_byState: ", len(claim_list_byState), claim_list_byState)
        print("claim_list_byCover: ", len(claim_list_byCover), claim_list_byCover)
        claim_list = (
            claim_list_byState
            if len(claim_list_byState) >= len(claim_list_byCover)
            else claim_list_byCover
        )
        print("claim_list: ", claim_list)

        claim_snakes_indexes = []

        # for agent claiming a bean safely, simply return its action
        for c in claim_list:
            if ctrl_agent_index == c[1]:  # the controlled agent claim a bean
                print(
                    "the controlled agent {} claim a bean {} within {} step(s)".format(
                        c[1], c[0], c[2]
                    )
                )
                action = [
                    bd.claim2action(
                        claim_position=bd.beans_positions[c[0]],
                        snake_idx=c[1],
                        step_count=c[2],
                        output_type="action",
                    )
                ]
                print("and play a action", action[0])
                print(
                    "*********************************** claim move"
                    " ******************************************"
                )
                return action
            claim_snakes_indexes.append(c[1])
        else:
            # not claim any beans,
            # traverse all possible action combination (at most 27),
            # choose one that claim most grids
            # calculate free team snakes indexes and safe positions list
            free_team_snakes_indexes = [
                _
                for _ in team_indexes
                if _ not in claim_snakes_indexes and _ not in defense_snakes_indexes
            ]
            safe_positions_list = []
            for idx in free_team_snakes_indexes:
                safe_positions = []  # may be empty list
                x_h, y_h = snakes[idx].head
                for x, y in [
                    ((x_h + 1) % bd.height, y_h),  # down
                    ((x_h - 1) % bd.height, y_h),  # up
                    (x_h, (y_h + 1) % bd.width),  # right
                    (x_h, (y_h - 1) % bd.width),
                ]:  # left
                    if bd.board[x][y] == idx + snakes_count:
                        safe_positions.append(
                            [x, y]
                        )  # should be further tested if exists breath!!
                safe_positions_list.append(safe_positions)

            # delete snakes whose safe positions are [], which means they are dying
            check_list = [_ != [] for _ in safe_positions_list]
            free_team_snakes_indexes = list(
                np.array(free_team_snakes_indexes)[check_list]
            )  # [idx1, idx2]
            safe_positions_list = [
                _ for _ in safe_positions_list if _
            ]  # [[pos1, pos2, pos3], [pos1, pos2, pos3]

            print("free_team_snakes_indexes: ", free_team_snakes_indexes)
            print("safe_positions_list: ", safe_positions_list)

            # create new snakes
            snakes_next = copy.deepcopy(snakes)
            for c in claim_list:  # claimed snake make one move
                idx = c[1]
                if idx in defense_snakes_indexes:
                    continue  # defense is prior to claim
                position = bd.claim2action(
                    claim_position=bd.beans_positions[c[0]],
                    snake_idx=idx,
                    step_count=c[2],
                    output_type="position",
                )
                snakes_next[idx].step(position)

            for c in defensive_claim_list:  # claimed snake make one move
                idx = c[1]
                position = bd.claim2action(
                    claim_position=c[0],
                    snake_idx=idx,
                    step_count=c[2],
                    output_type="position",
                )
                snakes_next[idx].step(position)

            # traverse and find the action combination with most grids claimed
            max_claimed_counts_sum = 0
            best_pos_comb = None
            for pos_comb in itertools.product(
                *safe_positions_list
            ):  # calculate cartesian product of safe positions list
                # initiate claimed_count
                for i, idx in enumerate(
                    free_team_snakes_indexes
                ):  # unclaimed and undead snake make one move
                    snakes_next[idx] = copy.deepcopy(snakes[idx])
                    snakes_next[idx].step(pos_comb[i])
                for snake in snakes_next.values():
                    snake.claimed_count = 0  # reset after deep copy !!
                bd_next = Board(
                    board_height, board_width, snakes_next, beans_positions, teams
                )
                with HiddenPrints():
                    while not all(
                        _ == [] for _ in bd_next.open.values()
                    ):  # loop until all values in open are empty list
                        bd_next.step()
                print(bd.board)
                claimed_counts = np.zeros(len(team_indexes))
                for i, idx in enumerate(
                    team_indexes
                ):  # not free, consider all team snakes!!
                    claimed_counts[i] = snakes_next[idx].claimed_count
                claimed_counts_sum = sum(claimed_counts)

                if claimed_counts_sum > max_claimed_counts_sum:
                    max_claimed_counts_sum = claimed_counts_sum
                    best_pos_comb = pos_comb  # one-to-one with free_team_snakes_indexes
                print("claimed_counts_sum: ", claimed_counts_sum)
                print("pos_comb: ", pos_comb)

            print(
                "max_claimed_counts_sum: ",
                max_claimed_counts_sum,
                "best_pos_comb: ",
                best_pos_comb,
            )

            if best_pos_comb:
                for i, idx in enumerate(free_team_snakes_indexes):
                    if ctrl_agent_index == idx:
                        action = [[0, 0, 0, 0]]
                        direction = snakes[idx].get_action(best_pos_comb[i])
                        action[0][direction] = 1
                        print(
                            "the controlled agent {} make a safe move".format(idx),
                            action[0],
                        )
                        print(
                            "*********************************** safe move"
                            " ******************************************"
                        )
                        return action

            # todo: design attack moves

            # no claim move, no safe move, no attack or die
            action = [[0, 0, 0, 0]]
            direction = snakes[ctrl_agent_index].legal_action[0]
            action[0][direction] = 1
            print(
                "the controlled agent {} play a random action and is dying".format(
                    ctrl_agent_index
                ),
                action[0],
            )
            print(
                "*********************************** random move"
                " ******************************************"
            )
            return action
