# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/7/30 17:24 下午
# 描述：
import itertools
import random
from itertools import count
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces
from PIL import Image, ImageDraw, ImageFont

from .discrete import Discrete
from .gridgame import GridGame, generate_color
from .observation import *


def convert_to_onehot(joint_action):
    new_joint_action = []
    for action in joint_action:
        onehot_action = np.zeros(4)
        onehot_action[action] = 1
        new_joint_action.append(onehot_action)
    return new_joint_action


conf_dict = {
    "snakes_1v1": {
        "class_literal": "SnakeEatBeans",
        "n_player": 2,
        "board_width": 8,
        "board_height": 6,
        "cell_range": 4,
        "n_beans": 5,
        "max_step": 50,
        "game_name": "snakes",
        "is_obs_continuous": False,
        "is_act_continuous": False,
        "agent_nums": [1, 1],
        "obs_type": ["dict", "dict"],
    },
    "snakes_3v3": {
        "class_literal": "SnakeEatBeans",
        "n_player": 6,
        "board_width": 20,
        "board_height": 10,
        "cell_range": 8,
        "n_beans": 5,
        "max_step": 200,
        "game_name": "snakes",
        "is_obs_continuous": False,
        "is_act_continuous": False,
        "agent_nums": [3, 3],
        "obs_type": ["dict", "dict"],
    },
}


class SnakeEatBeans(GridGame, GridObservation, DictObservation):
    def __init__(self, render_mode: Optional[str] = None, id: Optional[str] = None):
        assert id in conf_dict.keys(), f"id must be in {conf_dict.keys()}, but get {id}"
        conf = conf_dict[id]
        self.terminate_flg = False
        colors = conf.get("colors", [(255, 255, 255), (255, 140, 0)])
        super(SnakeEatBeans, self).__init__(conf, colors)
        # 0: 没有 1：食物 2-n_player+1:各玩家蛇身
        self.n_cell_type = self.n_player + 2
        self.step_cnt = 1
        self.n_beans = int(conf["n_beans"])
        # 方向[-2,2,-1,1]分别表示[上，下，左，右]
        self.actions = [-2, 2, -1, 1]
        self.actions_name = {-2: "up", 2: "down", -1: "left", 1: "right"}
        self.snakes_position = {}
        self.players = []
        self.cur_bean_num = 0
        self.beans_position = []
        # 1<= init_len <= 3
        self.init_len = 3
        self.current_state = self.init_state()
        self.all_observes = self.get_all_observes()
        if self.n_player * self.init_len > self.board_height * self.board_width:
            raise Exception(
                "玩家数量过多：%d，超出board范围：%d，%d"
                % (self.n_player, self.board_width, self.board_height)
            )

        self.input_dimension = self.board_width * self.board_height
        self.action_dim = self.get_action_dim()

        self.num_agents = conf["agent_nums"][0]
        self.num_enemys = conf["agent_nums"][1]

        self.observation_space = [
            spaces.Box(low=-np.inf, high=-np.inf, shape=(288,), dtype=np.float32)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(288,), dtype=np.float32)
        ]
        # self.action_space = [Discrete(4) for _ in range(self.n_player)]
        self.action_space = Discrete(4)

        self.episode = 0
        self.fig, self.ax = None, None
        if render_mode in ["human", "rgb_array"]:
            self.need_render = True
            if render_mode == "human":
                plt.ion()
                self.fig, self.ax = plt.subplots()
        else:
            self.need_render = False
        self.render_mode = render_mode
        self.img_list = None
        self.render_img = None

        self.init_colors = colors
        self.colors = None

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(0)
            random.seed(0)
        else:
            np.random.seed(seed)
            random.seed(seed)

    def check_win(self):
        flg = self.won.index(max(self.won)) + 2
        return flg

    def get_grid_observation(self, current_state, player_id, info_before):
        return current_state

    def get_dict_observation(self, current_state, player_id, info_before):
        key_info = {1: self.beans_position}
        for i in range(self.n_player):
            snake = self.players[i]
            key_info[snake.player_id] = snake.segments
        # key_info['state_map'] = current_state
        key_info["board_width"] = self.board_width
        key_info["board_height"] = self.board_height
        key_info["last_direction"] = (
            info_before.get("directions") if isinstance(info_before, dict) else None
        )
        key_info["controlled_snake_index"] = player_id

        return key_info

    def set_action_space(self):
        action_space = [[Discrete(4)] for _ in range(self.n_player)]
        return action_space

    def reset(self):
        if self.need_render:
            self.img_list = []
            self.render_img = None
        if self.colors is None:
            self.colors = (
                self.init_colors
                + generate_color(self.cell_size - len(self.init_colors) + 1)
                if self.init_colors is not None
                else generate_color(self.cell_size)
            )

        self.step_cnt = 1
        self.snakes_position = (
            {}
        )  # 格式类似于{1: [[3, 1], [4, 3], [1, 2], [0, 6], [3, 3]], 2: [[3, 0], [3, 7], [3, 6]], 3: [[2, 7], [1, 7], [0, 7]]}
        self.players = []
        self.cur_bean_num = 0
        self.beans_position = []
        self.current_state = self.init_state()
        self.all_observes = self.get_all_observes()
        self.terminate_flg = False

        self.episode += 1

        # available actions
        left_avail_actions = np.ones([self.num_agents, self.action_dim])
        right_avail_actions = np.ones([self.num_enemys, self.action_dim])
        avail_actions = np.concatenate([left_avail_actions, right_avail_actions], 0)
        # process obs

        info = {"action_mask": avail_actions}
        self.inner_render()

        return self.all_observes, info

    def step(self, joint_action):
        joint_action = convert_to_onehot(joint_action)

        joint_action = np.expand_dims(joint_action, 1)
        all_observes, info_after = self.get_next_state(joint_action)
        done = self.is_terminal()
        reward = self.get_reward(joint_action)
        left_avail_actions = np.ones([self.num_agents, self.action_dim])
        right_avail_actions = np.ones([self.num_enemys, self.action_dim])
        avail_actions = np.concatenate([left_avail_actions, right_avail_actions], 0)
        rewards = np.expand_dims(np.array(reward), axis=1)

        dones = [done] * self.n_player
        infos = info_after

        infos.update({"action_mask": avail_actions})
        self.inner_render()

        return all_observes, rewards, dones, infos

    # obs: 0-空白 1-豆子 2-我方蛇头 3-我方蛇身 4-敌方蛇头 5-敌方蛇身

    def inner_render(self):
        if not self.need_render:
            return
        img = self.render_board()
        self.render_img = img
        if self.render_mode == "human":
            self.ax.imshow(img, cmap="gray")
            plt.draw()
            plt.pause(0.1)

    def render(self):
        return self.render_img

    def raw2vec(self, raw_obs):
        control_index = raw_obs["controlled_snake_index"]
        width = raw_obs["board_width"]
        height = raw_obs["board_height"]
        beans = raw_obs[1]
        ally_pos = raw_obs[control_index]
        enemy_pos = raw_obs[5 - control_index]

        obs = np.zeros(width * height * self.n_player, dtype=int)
        ally_head_h, ally_head_w = ally_pos[0]
        enemy_head_h, enemy_head_w = enemy_pos[0]
        obs[ally_head_h * width + ally_head_w] = 2
        obs[height * width + ally_head_h * width + ally_head_w] = 4
        obs[enemy_head_h * width + enemy_head_w] = 4
        obs[height * width + enemy_head_h * width + enemy_head_w] = 2

        for bean in beans:
            h, w = bean
            obs[h * width + w] = 1
            obs[height * width + h * width + w] = 1

        for p in ally_pos[1:]:
            h, w = p
            obs[h * width + w] = 3
            obs[height * width + h * width + w] = 5

        for p in enemy_pos[1:]:
            h, w = p
            obs[h * width + w] = 5
            obs[height * width + h * width + w] = 3

        obs_ = np.array([])
        for i in obs:
            obs_ = np.concatenate([obs_, np.eye(6)[i]])
        obs_ = obs_.reshape(-1, width * height * 6)

        return obs_

    def init_state(self):
        for i in range(self.n_player):
            s = Snake(i + 2, self.board_width, self.board_height, self.init_len)
            s_len = 1
            while s_len < self.init_len:
                if s_len == 1 and i > 0:
                    origin_hit = self.is_hit(s.headPos, self.snakes_position)
                else:
                    origin_hit = 0
                cur_head = s.move_and_add(self.snakes_position)
                cur_hit = self.is_hit(cur_head, self.snakes_position) or self.is_hit(
                    cur_head, {i: s.segments[1:]}
                )
                if origin_hit or cur_hit:
                    x = random.randrange(0, self.board_height)
                    y = random.randrange(0, self.board_width)
                    s.headPos = [x, y]
                    s.segments = [s.headPos]
                    s.direction = random.choice(self.actions)
                    s_len = 1
                else:
                    s_len += 1
            self.snakes_position[s.player_id] = s.segments
            self.players.append(s)

        self.generate_beans()
        self.init_info = {
            "snakes_position": [
                list(v)
                for k, v in sorted(
                    self.snakes_position.items(), key=lambda item: item[0]
                )
            ],
            "beans_position": list(self.beans_position),
        }
        directs = []
        for i in range(len(self.players)):
            s = self.players[i]
            directs.append(self.actions_name[s.direction])
        self.init_info["directions"] = directs

        return self.update_state()

    def update_state(self):
        next_state = [
            [[0] * self.cell_dim for _ in range(self.board_width)]
            for _ in range(self.board_height)
        ]
        for i in range(self.n_player):
            snake = self.players[i]
            for pos in snake.segments:
                next_state[pos[0]][pos[1]][0] = i + 2

        for pos in self.beans_position:
            next_state[pos[0]][pos[1]][0] = 1

        return next_state

    def step_before_info(self, info=""):
        directs = []
        for i in range(len(self.players)):
            s = self.players[i]
            directs.append(self.actions_name[s.direction])
        info = {"directions": directs}

        return info

    def is_hit(self, cur_head, snakes_position):
        is_hit = False
        for k, v in snakes_position.items():
            for pos in v:
                if cur_head == pos:
                    is_hit = True
                    # print("hit:", cur_head, snakes_position)
                    break
            if is_hit:
                break

        return is_hit

    def generate_beans(self):
        all_valid_positions = set(
            itertools.product(range(0, self.board_height), range(0, self.board_width))
        )
        all_valid_positions = all_valid_positions - set(map(tuple, self.beans_position))
        for positions in self.snakes_position.values():
            all_valid_positions = all_valid_positions - set(map(tuple, positions))

        left_bean_num = self.n_beans - self.cur_bean_num
        all_valid_positions = np.array(list(all_valid_positions))
        left_valid_positions = len(all_valid_positions)

        new_bean_num = (
            left_bean_num
            if left_valid_positions > left_bean_num
            else left_valid_positions
        )

        if left_valid_positions > 0:
            new_bean_positions_idx = np.random.choice(
                left_valid_positions, size=new_bean_num, replace=False
            )
            new_bean_positions = all_valid_positions[new_bean_positions_idx]
        else:
            new_bean_positions = []

        for new_bean_pos in new_bean_positions:
            self.beans_position.append(list(new_bean_pos))
            self.cur_bean_num += 1

    def get_all_observes(self, before_info=""):
        self.all_observes = []
        for i in range(self.n_player):
            each_obs = self.get_dict_observation(self.current_state, i + 2, before_info)
            self.all_observes.append(each_obs)

        return self.all_observes

    def get_next_state(self, all_action):
        before_info = self.step_before_info()
        not_valid = self.is_not_valid_action(all_action)
        if not not_valid:
            # 各玩家行动
            # print("current_state", self.current_state)
            eat_snakes = [0] * self.n_player
            others_reward = [
                0
            ] * self.n_player  # 记录对方获得的奖励，因为是零和博弈，所以敌人获得了多少奖励，我方就要减去多少奖励
            for i in range(self.n_player):  # 判断是否吃到了豆子
                snake = self.players[i]
                act = self.actions[np.argmax(all_action[i][0])]
                # print(snake.player_id, "此轮的动作为：", self.actions_name[act])
                snake.change_direction(act)
                snake.move_and_add(self.snakes_position)  # 更新snake.segment
                if self.be_eaten(snake.headPos):  # @yanxue
                    snake.snake_reward = 1
                    eat_snakes[i] = 1
                else:
                    snake.snake_reward = 0
                    snake.pop()
                # print(snake.player_id, snake.segments)   # @yanxue
            snake_position = [[-1] * self.board_width for _ in range(self.board_height)]
            re_generatelist = [0] * self.n_player
            for i in range(self.n_player):  # 判断是否相撞
                snake = self.players[i]
                segment = snake.segments
                for j in range(len(segment)):
                    x = segment[j][0]
                    y = segment[j][1]
                    if snake_position[x][y] != -1:
                        if j == 0:  # 撞头
                            re_generatelist[i] = 1
                        compare_snake = self.players[snake_position[x][y]]
                        if [x, y] == compare_snake.segments[0]:  # 两头相撞won
                            re_generatelist[snake_position[x][y]] = 1
                    else:
                        snake_position[x][y] = i
            for i in range(self.n_player):
                snake = self.players[i]
                if re_generatelist[i] == 1:
                    if eat_snakes[i] == 1:
                        snake.snake_reward = (
                            self.init_len - len(snake.segments) + 1
                        )  # 身体越长，惩罚越大
                    else:
                        snake.snake_reward = self.init_len - len(snake.segments)
                    snake.segments = []
            for i in range(self.num_agents):
                others_reward[self.num_agents :] = [
                    others_reward[j + self.num_agents] + self.players[i].snake_reward
                    for j in range(self.num_enemys)
                ]
                others_reward[self.num_agents :] = [
                    others_reward[j + self.num_agents] // self.num_agents
                    for j in range(self.num_enemys)
                ]
            for i in range(self.num_enemys):
                others_reward[: self.num_agents] = [
                    others_reward[j] + self.players[i + self.num_agents].snake_reward
                    for j in range(self.num_agents)
                ]
                others_reward[: self.num_agents] = [
                    others_reward[j] // self.num_enemys for j in range(self.num_agents)
                ]
            for i in range(self.n_player):
                self.players[i].snake_reward -= others_reward[i]
            for i in range(self.n_player):
                snake = self.players[i]
                if re_generatelist[i] == 1:
                    snake = self.clear_or_regenerate(snake)
                self.snakes_position[snake.player_id] = snake.segments
                snake.score = snake.get_score()
            # yanxue add
            # 更新状态
            self.generate_beans()

            next_state = self.update_state()
            self.current_state = next_state
            self.step_cnt += 1

            self.won = [0] * self.n_player

            for i in range(self.n_player):
                s = self.players[i]
                self.won[i] = s.score
            info_after = {}
            info_after["snakes_position"] = [
                list(v)
                for k, v in sorted(
                    self.snakes_position.items(), key=lambda item: item[0]
                )
            ]
            info_after["beans_position"] = list(self.beans_position)
            info_after["hit"] = re_generatelist
            info_after["score"] = self.won
            self.all_observes = self.get_all_observes(before_info)

            return self.all_observes, info_after

    def clear_or_regenerate(self, snake):
        direct_x = [0, 1, -1, 0]
        direct_y = [1, 0, 0, -1]
        snake.segments = []
        snake.score = 0
        grid = self.get_render_data(self.update_state())

        def can_regenerate():
            for x in range(self.board_height):
                for y in range(self.board_width):
                    if grid[x][y] == 0:
                        q = []
                        q.append([x, y])
                        seg = []
                        while q:
                            cur = q.pop(0)
                            if cur not in seg:
                                seg.append(cur)
                            for i in range(4):
                                nx = (direct_x[i] + cur[0]) % self.board_height
                                ny = (direct_y[i] + cur[1]) % self.board_width
                                # if nx < 0 or nx >= self.board_height or ny < 0 or ny >= self.board_width:
                                #     continue
                                if grid[nx][ny] == 0 and [nx, ny] not in q:
                                    grid[nx][ny] = 1
                                    q.append([nx, ny])
                            if len(seg) == self.init_len:
                                # print("regenerate")
                                if len(seg) < 3:
                                    snake.direction = random.choice(self.actions)
                                elif len(seg) == 3:
                                    mid = (
                                        [seg[1][0], seg[2][1]],
                                        [seg[2][0], seg[1][1]],
                                    )
                                    if seg[0] in mid:
                                        seg[0], seg[1] = seg[1], seg[0]
                                    snake.segments = seg
                                    snake.headPos = seg[0]
                                    if seg[0][0] == seg[1][0]:
                                        # 右
                                        if seg[0][1] > seg[1][1]:
                                            snake.direction = 1
                                        # 左
                                        else:
                                            snake.direction = -1
                                    elif seg[0][1] == seg[1][1]:
                                        # 下
                                        if seg[0][0] > seg[1][0]:
                                            snake.direction = 2
                                        # 上
                                        else:
                                            snake.direction = -2
                                # print("re head", snake.headPos)  # 输出重新生成的蛇
                                # print("re snakes segments", snake.segments)
                                return True
            # print("clear")
            return False

        flg = can_regenerate()
        if not flg:
            self.terminate_flg = True
        # print(self.terminate_flg)
        return snake

    # def is_not_valid_action(self, joint_action):
    #     not_valid = 0
    #     if len(joint_action) != self.n_player:
    #         raise Exception("joint action 维度不正确！", len(joint_action))
    #
    #     for i in range(len(joint_action)):
    #         if len(joint_action[i][0]) != 4:
    #             raise Exception("玩家%d joint action维度不正确！" % i, joint_action[i])
    #     return not_valid

    def is_not_valid_action(self, all_action):
        not_valid = 0
        if len(all_action) != self.n_player:
            raise Exception("all action dimension is wrong!", len(all_action))

        for i in range(self.n_player):
            if len(all_action[i][0]) != 4:
                raise Exception(
                    "Player %d has wrong joint action dimension!" % i, all_action[i]
                )
        return not_valid

    def get_reward(self, all_action):
        r = [0] * self.n_player
        for i in range(self.n_player):
            r[i] = self.players[i].snake_reward
            self.n_return[i] += r[i]
        # print("score:", self.won)
        return r

    def is_terminal(self):
        all_member = self.n_beans
        # all_member = len(self.beans_position)
        for s in self.players:
            all_member += len(s.segments)
        is_done = (
            self.step_cnt > self.max_step
            or all_member > self.board_height * self.board_width
        )

        return is_done or self.terminate_flg

    def encode(self, actions):
        joint_action = self.init_action_space()
        if len(actions) != self.n_player:
            raise Exception("action输入维度不正确！", len(actions))
        for i in range(self.n_player):
            joint_action[i][0][int(actions[i])] = 1
        return joint_action

    def get_terminal_actions(self):
        print("请输入%d个玩家的动作方向[0-3](上下左右)，空格隔开：" % self.n_player)
        cur = input()
        actions = cur.split(" ")
        return self.encode(actions)

    def be_eaten(self, snake_pos):
        for bean in self.beans_position:
            if snake_pos[0] == bean[0] and snake_pos[1] == bean[1]:
                self.beans_position.remove(bean)
                self.cur_bean_num -= 1
                return True
        return False

    def get_action_dim(self):
        action_dim = 1
        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim

    def draw_board(self):
        cols = [chr(i) for i in range(65, 65 + self.board_width)]
        s = ", ".join(cols)
        print("  ", s)
        for i in range(self.board_height):
            # print(i)
            print(chr(i + 65), self.current_state[i])

    @staticmethod
    def _render_board(state, board, colors, unit, fix, extra_info):
        im = GridGame._render_board(state, board, colors, unit, fix)
        draw = ImageDraw.Draw(im)
        # fnt = ImageFont.truetype("Courier.dfont", 16)
        fnt = ImageFont.load_default()
        for i, pos in zip(count(1), extra_info):
            x, y = pos
            draw.text(
                ((y + 1 / 4) * unit, (x + 1 / 4) * unit),
                "#{}".format(i),
                font=fnt,
                fill=(0, 0, 0),
            )

        return im

    def render_board(self):
        extra_info = [tuple(x.headPos) for x in self.players]
        im_data = np.array(
            SnakeEatBeans._render_board(
                self.get_render_data(self.current_state),
                self.grid,
                self.colors,
                self.grid_unit,
                self.grid_unit_fix,
                extra_info,
            )
        )
        return im_data

    @staticmethod
    def parse_extra_info(data):
        # return eval(re.search(r'({.*})', data['info_after']).group(1)).values()
        # d = (eval(eval(data)['snakes_position']).values())
        if isinstance(data, str):
            d = eval(data)["snakes_position"]
        else:
            d = data["snakes_position"]

        return [i[0] for i in d]

    def close(self):
        if self.render_mode == "human":
            plt.close(self.fig)


class Snake:
    def __init__(self, player_id, board_width, board_height, init_len):
        self.actions = [-2, 2, -1, 1]
        self.actions_name = {-2: "up", 2: "down", -1: "left", 1: "right"}
        self.direction = random.choice(self.actions)  # 方向[-2,2,-1,1]分别表示[上，下，左，右]
        self.board_width = board_width
        self.board_height = board_height
        x = random.randrange(0, board_height)
        y = random.randrange(0, board_width)
        self.segments = [[x, y]]
        self.headPos = self.segments[0]
        self.player_id = player_id
        self.score = 0
        self.snake_reward = 0
        self.init_len = init_len

    def get_score(self):
        return len(self.segments) - self.init_len

    def change_direction(self, act):
        if act + self.direction != 0:
            self.direction = act
        else:
            n_direct = random.choice(self.actions)
            while n_direct + self.direction == 0:
                n_direct = random.choice(self.actions)
            self.direction = n_direct
        #     print("方向不合法，重新生成")
        # print("direction", self.actions_name[self.direction])

    # 超过边界，可以穿越
    def update_position(self, position):
        position[0] %= self.board_height
        position[1] %= self.board_width
        return position

    def move_and_add(self, snakes_position):
        cur_head = list(self.headPos)
        # 根据方向移动蛇头的坐标
        #     右
        if self.direction == 1:
            cur_head[1] += 1
        #     左
        if self.direction == -1:
            cur_head[1] -= 1
        #     上
        if self.direction == -2:
            cur_head[0] -= 1
        #     下
        if self.direction == 2:
            cur_head[0] += 1

        cur_head = self.update_position(cur_head)
        # print("cur head", cur_head)
        # print("cur snakes positions", snakes_position)

        self.segments.insert(0, cur_head)
        self.headPos = self.segments[0]
        return cur_head

    def pop(self):
        self.segments.pop()  # 在蛇尾减去一格
