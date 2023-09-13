# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/11/13 3:51 下午
# 描述：observation的各种接口类
obs_type = ["grid", "vector", "dict"]


class GridObservation(object):
    def get_grid_observation(self, current_state, player_id, info_before):
        raise NotImplementedError

    def get_grid_many_observation(self, current_state, player_id_list, info_before=""):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_grid_observation(current_state, i, info_before))
        return all_obs


class VectorObservation(object):
    def get_vector_observation(self, current_state, player_id, info_before):
        raise NotImplementedError

    def get_vector_many_observation(
        self, current_state, player_id_list, info_before=""
    ):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_vector_observation(current_state, i, info_before))
        return all_obs


class DictObservation(object):
    def get_dict_observation(self, current_state, player_id, info_before):
        raise NotImplementedError

    def get_dict_many_observation(self, current_state, player_id_list, info_before=""):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_dict_observation(current_state, i, info_before))
        return all_obs


# todo: observation builder
class CustomObservation(object):
    def get_custom_observation(self, current_state, player_id):
        raise NotImplementedError

    def get_custom_obs_space(self, player_id):
        raise NotImplementedError

    def get_custom_many_observation(self, current_state, player_id_list):
        all_obs = []
        for i in player_id_list:
            all_obs.append(self.get_custom_observation(current_state, i))
        return all_obs

    def get_custom_many_obs_space(self, player_id_list):
        all_obs_space = []
        for i in player_id_list:
            all_obs_space.append(self.get_custom_obs_space(i))
        return all_obs_space
