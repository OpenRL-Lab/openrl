from typing import Any, Dict, List, Union

import numpy as np

class BaseReward(object):
    def __init__(self):
        self.step_reward_fn = dict()
        self.inner_reward_fn = dict()
        self.batch_reward_fn = dict()


    def step_reward(
            self, 
            data: Dict[str, Any]
        ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        rewards = data["reward"].copy()
        infos = []
        
        for rew_func in self.step_rew_funcs.values():
            new_rew, new_info = rew_func(data)
            if len(infos) == 0:
                infos = new_info
            else:
                for i in range(len(infos)):
                    infos[i].update(new_info[i])
            rewards += new_rew

        return rewards, infos

    def batch_rewards(self, buffer: Any) -> Dict[str, Any]:
        
        infos = dict()
        
        for rew_func in self.batch_rew_funcs.values():
            new_rew, new_info = rew_func()
            if len(infos) == 0:
                infos = new_info
            else:
                infos.update(new_info)
        # update rewards, and infos here
        
        return dict()
