from abc import abstractmethod


class BaseSelfplayStrategy:
    @abstractmethod
    def __init__(self, all_args, nenvs, exist_enemy_num):
        raise NotImplementedError

    @abstractmethod
    def getcnt(self):
        raise NotImplementedError

    @abstractmethod
    def update_enemy_ids(self, new_enemy_ids):
        raise NotImplementedError

    @abstractmethod
    def restore(self, model_dir):
        raise NotImplementedError

    @abstractmethod
    def get_qlist(self):
        raise NotImplementedError

    @abstractmethod
    def update_weight(self, enemy_loses):
        raise NotImplementedError

    @abstractmethod
    def update_win_rate(self, dones, enemy_wins):
        raise NotImplementedError

    @abstractmethod
    def push_newone(self):
        raise NotImplementedError

    @abstractmethod
    def get_plist(self):
        raise NotImplementedError
