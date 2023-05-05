from typing import Any, Dict, List, Union

import numpy as np


class BaseReward(object):
    def __init__(self):
        self.step_reward_fn = dict()
        self.inner_reward_fn = dict()
        self.batch_reward_fn = dict()

    def step_reward(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        rewards = data["rewards"].copy()
        infos = [dict() for _ in range(rewards.shape[0])]

        return rewards, infos

    def batch_rewards(self, buffer: Any) -> Dict[str, Any]:
        infos = dict()

        return infos
