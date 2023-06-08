import time
from typing import Any, Dict

import numpy as np

from openrl.envs.vec_env.vec_info.base_vec_info import BaseVecInfo


class SimpleVecInfo(BaseVecInfo):
    def __init__(self, parallel_env_num: int, agent_num: int):
        super().__init__(parallel_env_num, agent_num)

        self.infos = []

        self.start_time = time.time()
        self.total_step = 0

    def statistics(self, buffer: Any) -> Dict[str, Any]:
        # this function should be called each episode
        rewards = buffer.data.rewards.copy()
        self.total_step += np.prod(rewards.shape[:2])
        rewards = rewards.transpose(2, 1, 0, 3)
        info_dict = {}
        ep_rewards = []
        for i in range(self.agent_num):
            agent_reward = rewards[i].mean(0).sum()
            ep_rewards.append(agent_reward)
            info_dict["agent_{}/rollout_episode_reward".format(i)] = agent_reward

        info_dict["FPS"] = int(self.total_step / (time.time() - self.start_time))
        info_dict["rollout_episode_reward"] = np.mean(ep_rewards)
        return info_dict

    def append(self, info: Dict[str, Any]) -> None:
        self.infos.append(info)

    def reset(self) -> None:
        self.infos = []
        self.rewards = []
