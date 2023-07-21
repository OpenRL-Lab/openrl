import json

import numpy as np

from openrl.selfplay.strategies.base_strategy import BaseSelfplayStrategy


class SelfplayStrategy(BaseSelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        # qlist和history_cnt的数据结构
        self.all_args = all_args
        self.qlist = []
        self.history_cnt = 0
        self.enemy_ids = [0] * nenvs
        self.length = nenvs

    def getcnt(self):
        return self.history_cnt

    def update_enemy_ids(self, new_enemy_ids):
        self.enemy_ids = new_enemy_ids

    def restore(self, model_dir):
        with open(model_dir + "/enemy_history_info.json") as f_obj:
            enemy_info = json.load(f_obj)
        self.qlist = enemy_info["qlist"]
        self.history_cnt = enemy_info["history_cnt"]

    def get_qlist(self):
        return self.qlist

    def update_weight(self, enemy_loses):
        pass

    def update_win_rate(self, dones, enemy_wins):
        pass

    def push_newone(self):
        pass


class RatioSelfplayStrategy(SelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(RatioSelfplayStrategy, self).__init__(all_args, nenvs)

    def push_newone(self):
        self.history_cnt += 1

    def get_plist(self):
        if self.history_cnt == 1:
            return [1]
        temp_plist = np.logspace(
            0, self.history_cnt - 1, self.history_cnt, endpoint=True, base=1.5
        )
        temp_plist[-1] = sum(temp_plist[:-1]) * 4
        temp_plist /= sum(temp_plist)
        return temp_plist


class NaiveSelfplayStrategy(SelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(NaiveSelfplayStrategy, self).__init__(all_args, nenvs, exist_enemy_num)

    def push_newone(self):
        self.history_cnt += 1

    def get_plist(self):
        return [1] * (self.history_cnt - 1) + [4 * (self.history_cnt - 1)]

    def save_new_one(self):
        return True


class OnlyLatestSelfplayStrategy(SelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(OnlyLatestSelfplayStrategy, self).__init__(
            all_args, nenvs, exist_enemy_num
        )
        self.play_list = []
        self.max_play_num = all_args.max_play_num
        self.least_win_rate = all_args.least_win_rate

    def push_newone(self):
        self.play_list.append([])
        self.history_cnt += 1

    def get_plist(self):
        return [0] * (self.history_cnt - 1) + [1]

    def save_new_one(self, least_win_rate):
        if sum(np.array(self.play_list[-1]) == -1) >= least_win_rate * (
            len(self.play_list[-1]) + 1
        ) and len(self.play_list[-1]) >= (self.max_play_num - 10):
            return True

    def update_play_list(self, win_enemy_ids, tie_enemy_ids, lose_enemy_ids):
        for win_enemy_id in win_enemy_ids:
            self.play_list[win_enemy_id].append(1)
        for tie_enemy_id in tie_enemy_ids:
            self.play_list[tie_enemy_id].append(0)
        for lose_enemy_id in lose_enemy_ids:
            self.play_list[lose_enemy_id].append(-1)
        self.cut_overflow()

    def update_win_rate(self, enemy_wins, enemy_ties, enemy_loses):
        win_enemy_ids = np.array(self.enemy_ids)[enemy_wins]
        tie_enemy_ids = np.array(self.enemy_ids)[enemy_ties]
        lose_enemy_ids = np.array(self.enemy_ids)[enemy_loses]
        self.update_play_list(win_enemy_ids, tie_enemy_ids, lose_enemy_ids)

    def cut_overflow(self):
        for index in range(len(self.play_list)):
            if len(self.play_list[index]) > self.max_play_num:
                self.play_list[index] = self.play_list[index][
                    (-1) * self.max_play_num :
                ]

    def get_info_list(self, info_list):
        return_info = []
        for info in info_list:
            if info == "win":
                equal_num = 1
            elif info == "tie":
                equal_num = 0
            elif info == "lose":
                equal_num = -1
            num_list = []
            for enemy_play_list in self.play_list:
                if info == "play":
                    num_list.append(len(enemy_play_list))
                else:
                    num_list.append(int(sum(np.array(enemy_play_list) == equal_num)))
            return_info.append(num_list)
        return tuple(return_info)

    def get_enemy_play_dict(self):
        win_num_list, tie_num_list, lose_num_list, play_num_list = self.get_info_list(
            ["win", "tie", "lose", "play"]
        )
        return {
            "win_num_list": list(win_num_list),
            "tie_num_list": list(tie_num_list),
            "lose_num_list": list(lose_num_list),
            "play_num_list": list(play_num_list),
        }


class WeightSelfplayStrategy(SelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(WeightSelfplayStrategy, self).__init__(all_args, nenvs, exist_enemy_num)
        self.recent_weight = 0.8
        self.recent_num = 3
        self.gama = 1 / (nenvs)

    def push_newone(self):
        self.history_cnt += 1
        if self.history_cnt <= self.recent_num:
            return
        elif self.history_cnt == self.recent_num + 1:
            self.qlist = [1]
        else:
            self.qlist.append(max(self.qlist))

    def get_plist(self):
        temp_plist = np.zeros([self.history_cnt])
        temp_plist[: (-1 * self.recent_num)] = (
            np.exp(self.qlist) / sum(np.exp(self.qlist)) * (1 - self.recent_weight)
        )
        temp_plist[(-1 * self.recent_num) :] = self.recent_weight / self.recent_num
        return temp_plist

    def update_weight(self, enemy_loses):
        if self.history_cnt < self.recent_num + 2:
            return
        lose_enemy_ids = np.array(self.enemy_ids)[
            enemy_loses
        ]  # 输了的enemy_ids,进行更新,其中可能有重复的enemy_id
        for enemy_id in lose_enemy_ids:
            if enemy_id <= len(self.qlist) - 1:
                divide_num = (
                    len(self.qlist)
                    * np.exp(self.qlist[enemy_id])
                    / sum(np.exp(self.qlist))
                )
                next_weight = self.qlist[enemy_id] - self.gama / divide_num
                self.qlist[enemy_id] = next_weight


class WinRateSelfplayStrategy(SelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(WinRateSelfplayStrategy, self).__init__(all_args, nenvs, exist_enemy_num)
        self.max_play_num = all_args.max_play_num
        self.play_list = (
            []
        )  # 在该list中，每个对手维护一个长度不超过max_play_num的列表，1为该对手获胜, 0为平, -1为我方获胜
        self.recent_list = []
        self.recent_list_max_len = all_args.recent_list_max_len
        self.latest_weight = all_args.latest_weight
        self.least_win_rate = all_args.least_win_rate
        self.stage2_least_win_rate = all_args.least_win_rate
        self.stage = 1
        self.newest_pos = all_args.newest_pos
        self.newest_weight = all_args.newest_weight

    def push_newone(self):
        self.play_list.append([])
        self.history_cnt += 1

    def get_info_list(self, info_list):
        return_info = []
        for info in info_list:
            if info == "win":
                equal_num = 1
            elif info == "tie":
                equal_num = 0
            elif info == "lose":
                equal_num = -1
            num_list = []
            for enemy_play_list in self.play_list:
                if info == "play":
                    num_list.append(len(enemy_play_list))
                else:
                    num_list.append(int(sum(np.array(enemy_play_list) == equal_num)))
            return_info.append(num_list)
        return tuple(return_info)

    def get_plist(self):
        def f_hard(win_rate_list):
            p = 1
            return win_rate_list**p

        def f_var(win_rate_list):
            return (1 - win_rate_list) * win_rate_list

        win_num_list, tie_num_list, play_num_list = self.get_info_list(
            ["win", "tie", "play"]
        )
        win_rate_list = (
            np.array(win_num_list) + 0.5 * np.array(tie_num_list) + 0.5
        ) / (np.array(play_num_list) + 1)
        return f_hard(win_rate_list)

    def update_play_list(self, win_enemy_ids, tie_enemy_ids, lose_enemy_ids):
        if self.stage == 2:
            win_enemy_num = (np.array(win_enemy_ids) != self.newest_pos).sum()
            tie_enemy_num = (np.array(tie_enemy_ids) != self.newest_pos).sum()
            lose_enemy_num = (np.array(lose_enemy_ids) != self.newest_pos).sum()
            self.recent_list += (
                [1] * win_enemy_num + [0] * tie_enemy_num + [-1] * lose_enemy_num
            )
        for win_enemy_id in win_enemy_ids:
            self.play_list[win_enemy_id].append(1)
        for tie_enemy_id in tie_enemy_ids:
            self.play_list[tie_enemy_id].append(0)
        for lose_enemy_id in lose_enemy_ids:
            self.play_list[lose_enemy_id].append(-1)
        self.cut_overflow()

    def update_win_rate(self, enemy_wins, enemy_ties, enemy_loses):
        win_enemy_ids = np.array(self.enemy_ids)[enemy_wins]
        tie_enemy_ids = np.array(self.enemy_ids)[enemy_ties]
        lose_enemy_ids = np.array(self.enemy_ids)[enemy_loses]
        self.update_play_list(win_enemy_ids, tie_enemy_ids, lose_enemy_ids)

    def restore(self, model_dir):
        with open(model_dir + "/enemy_history_info.json") as f_obj:
            enemy_info = json.load(f_obj)
        self.history_cnt = enemy_info["history_cnt"]
        self.play_list = enemy_info["play_list"]

    def get_enemy_play_dict(self):
        win_num_list, tie_num_list, lose_num_list, play_num_list = self.get_info_list(
            ["win", "tie", "lose", "play"]
        )
        return {
            "win_num_list": list(win_num_list),
            "tie_num_list": list(tie_num_list),
            "lose_num_list": list(lose_num_list),
            "play_num_list": list(play_num_list),
        }

    def update_win_info(self, data):
        win_enemy_ids, tie_enemy_ids, lose_enemy_ids = (
            data["win_enemy_ids"],
            data["tie_enemy_ids"],
            data["lose_enemy_ids"],
        )
        self.update_play_list(win_enemy_ids, tie_enemy_ids, lose_enemy_ids)

    def cut_overflow(self):
        for index in range(len(self.play_list)):
            if len(self.play_list[index]) > self.max_play_num:
                self.play_list[index] = self.play_list[index][
                    (-1) * self.max_play_num :
                ]
        if len(self.recent_list) > self.recent_list_max_len:
            self.recent_list = self.recent_list[(-1) * self.recent_list_max_len :]

    def save_new_one(self, least_win_rate):
        if self.stage == 1:
            if sum(np.array(self.play_list[-1]) == -1) >= least_win_rate * (
                len(self.play_list[-1]) + 1
            ) and len(self.play_list[-1]) >= (self.max_play_num - 10):
                if self.getcnt() - self.all_args.exist_enemy_num == 1:
                    return True
                self.stage = 2
                print("switch to stage 2")
        if self.stage == 2:
            if sum(np.array(self.recent_list) == -1) >= self.stage2_least_win_rate * (
                len(self.recent_list) + 1
            ) and len(self.recent_list) >= (self.recent_list_max_len - 10):
                self.stage = 1
                self.recent_list = []
                return True
        return False


class ExistEnemySelfplayStrategy(WinRateSelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(ExistEnemySelfplayStrategy, self).__init__(
            all_args, nenvs, exist_enemy_num
        )
        self.all_args = all_args
        self.enemy_ids = [0] * nenvs  # 第一个step就会更新，所以初始化无所谓
        # 列表的前exist_enemy_num个为已存在的对手
        if exist_enemy_num > 0:
            self.play_list = [[]] * exist_enemy_num
        self.history_cnt = exist_enemy_num
        self.exist_enemy_num = exist_enemy_num
        self.max_enemy_num = all_args.max_enemy_num

    def get_final_plist(self, f_hard, f_var):
        raise NotImplementedError

    def get_plist(self):
        def f_hard(win_rate_list):
            p = 2
            return win_rate_list**p

        def f_var(win_rate_list):
            return (1 - win_rate_list) * win_rate_list

        plist = self.get_final_plist(f_hard, f_var)
        if self.max_enemy_num != -1:
            if self.history_cnt - self.exist_enemy_num > self.max_enemy_num:
                mask_index = np.array(
                    list(
                        range(
                            self.exist_enemy_num, self.history_cnt - self.max_enemy_num
                        )
                    )
                )
                zero_vec = np.zeros(
                    self.history_cnt - self.exist_enemy_num - self.max_enemy_num
                )
                plist[mask_index] = zero_vec

        return plist


class VarExistEnemySelfplayStrategy(ExistEnemySelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(VarExistEnemySelfplayStrategy, self).__init__(
            all_args, nenvs, exist_enemy_num
        )

    def get_final_plist(self, f_hard, f_var):
        win_num_list, tie_num_list, play_num_list = self.get_info_list(
            ["win", "tie", "play"]
        )
        win_rate_list = (
            np.array(win_num_list) + 0.5 * np.array(tie_num_list) + 0.5
        ) / (np.array(play_num_list) + 1)
        win_rate_list = f_var(win_rate_list)

        return win_rate_list


class WeightExistEnemySelfplayStrategy(ExistEnemySelfplayStrategy):
    def __init__(self, all_args, nenvs, exist_enemy_num):
        super(WeightExistEnemySelfplayStrategy, self).__init__(
            all_args, nenvs, exist_enemy_num
        )

    def get_final_plist(self, f_hard, f_var):
        win_num_list, tie_num_list, play_num_list = self.get_info_list(
            ["win", "tie", "play"]
        )
        win_rate_list = (
            np.array(win_num_list) + 0.5 * np.array(tie_num_list) + 0.5
        ) / (np.array(play_num_list) + 1)

        if self.stage == 1:
            win_rate_list = f_hard(win_rate_list)[:-1]
            # if self.newest_pos != -1:
            #     win_rate_list[self.newest_pos] = 0
            win_rate_list = (
                win_rate_list / (sum(win_rate_list) + 1e-8) * (1 - self.latest_weight)
            )
            return list(win_rate_list) + [self.latest_weight]
        elif self.stage == 2:
            win_rate_list = f_hard(win_rate_list)
            if self.newest_pos != -1:
                win_rate_list[self.newest_pos] = self.newest_weight
                index_without_newest = list(range(self.history_cnt))
                index_without_newest.remove(self.newest_pos)
                win_rate_list[index_without_newest] /= sum(
                    win_rate_list[index_without_newest]
                )
                win_rate_list[index_without_newest] *= 1 - self.newest_weight
            else:
                win_rate_list /= sum(win_rate_list)
            return win_rate_list
