import time
from typing import Any, Dict

import numpy as np


class NLPVecInfo:
    def __init__(self, parallel_env_num, agent_num):
        self.parallel_env_num = parallel_env_num
        self.agent_num = agent_num

        self.episode_infos = []
        self.rewards = []

        self.start_time = time.time()
        self.total_step = 0

        self.log_items = [
            "alpha",
            "kl_div",
        ]
        self.log_final_items = [
            "intent_rewards",
            "meteor_rewards",
        ]

    def statistics(self) -> Dict[str, Any]:
        # this function should be called each episode
        rewards = np.array(self.rewards)
        self.total_step += np.prod(rewards.shape[:2])
        rewards = rewards.transpose(2, 1, 0, 3)
        info_dict = {}
        ep_rewards = []
        for i in range(self.agent_num):
            agent_reward = rewards[i].mean(0).sum()
            ep_rewards.append(agent_reward)
            info_dict["agent_{}/episode_reward".format(i)] = agent_reward

        new_infos = dict()
        for step_infos in self.episode_infos:
            for env_infos in step_infos:
                for key in env_infos:
                    if key in new_infos:
                        new_infos[key].append(env_infos[key])
                    else:
                        new_infos[key] = [env_infos[key]]
        for key in new_infos:
            if key in self.log_items:
                info_dict[key] = np.array(new_infos[key]).mean()
        for key in self.log_final_items:
            if "final_info" in new_infos:
                info_dict[key] = np.array(
                    [item[key] for item in new_infos["final_info"]]
                ).mean()

        info_dict["FPS"] = int(self.total_step / (time.time() - self.start_time))
        info_dict["episode_reward"] = np.mean(ep_rewards)

        return info_dict

    def append(self, reward, info):
        assert reward.shape[:2] == (self.parallel_env_num, self.agent_num)
        self.episode_infos.append(info)
        self.rewards.append(reward)

    def reset(self):
        self.episode_infos = []
        self.rewards = []


class VecInfo:
    def __init__(self, parallel_env_num, agent_num):
        self.parallel_env_num = parallel_env_num
        self.agent_num = agent_num

        self.infos = []
        self.rewards = []

        self.start_time = time.time()
        self.total_step = 0

    def statistics(self) -> Dict[str, Any]:
        # this function should be called each episode
        rewards = np.array(self.rewards)
        self.total_step += np.prod(rewards.shape[:2])
        rewards = rewards.transpose(2, 1, 0, 3)
        info_dict = {}
        ep_rewards = []
        for i in range(self.agent_num):
            agent_reward = rewards[i].mean(0).sum()
            ep_rewards.append(agent_reward)
            info_dict["agent_{}/episode_reward".format(i)] = agent_reward

        info_dict["FPS"] = int(self.total_step / (time.time() - self.start_time))
        info_dict["episode_reward"] = np.mean(ep_rewards)
        return info_dict

    def append(self, reward, info):
        assert reward.shape[:2] == (self.parallel_env_num, self.agent_num)
        self.infos.append(info)
        self.rewards.append(reward)

    def reset(self):
        self.infos = []
        self.rewards = []
