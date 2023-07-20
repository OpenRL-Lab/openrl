from typing import Any, Dict, List, Tuple

import numpy as np
from torch import nn

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.rewards.base_reward import BaseReward


class RewardPredictor:
    def __init__(
        self,
        cfg,
        discriminator: nn.Module,
    ):
        self.discriminator = discriminator
        self.gamma = cfg.gamma
        self.update_rms = False

    def __call__(self, data):
        step = data["step"]
        obs = data["buffer"].data.critic_obs[step]
        action = data["actions"]
        mask = data["buffer"].data.masks[step]
        reward = self.discriminator.predict_reward(
            obs,
            action,
            self.gamma,
            mask,
            update_rms=self.update_rms,
        )
        return reward, {}


class GAILReward(BaseReward):
    def __init__(self, env: BaseVecEnv):
        super().__init__(env)

    def set_discriminator(self, cfg, discriminator: nn.Module):
        self.step_rew_funcs = {
            "gail_discriminator": RewardPredictor(cfg, discriminator),
        }

    def step_reward(
        self, data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # step reward
        infos = []
        # rewards = data["rewards"].copy()
        rewards = None
        for rew_func in self.step_rew_funcs.values():
            new_rew, new_info = rew_func(data)
            if len(infos) == 0:
                infos = new_info
            else:
                for i in range(len(infos)):
                    infos[i].update(new_info[i])
            # rewards += new_rew
            rewards = new_rew

        return rewards, infos
