from typing import Any, Dict, List, Union

import numpy as np

from openrl.envs.vec_env.base_venv import BaseVecEnv


class BaseReward(object):
    def __init__(self, env: BaseVecEnv):
        self.step_rew_funcs = dict()
        self.inner_rew_funcs = dict()
        self.batch_rew_funcs = dict()

    def step_reward(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        rewards = data["rewards"].copy()
        infos = [dict() for _ in range(rewards.shape[0])]

        return rewards, infos

    def batch_rewards(self, buffer: Any) -> Dict[str, Any]:
        infos = dict()

        return infos
