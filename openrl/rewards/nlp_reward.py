from typing import Any, Dict

import numpy as np
from gymnasium import Env

from openrl.buffers.normal_buffer import NormalReplayBuffer
from openrl.envs.nlp.nlp_rewards import IntentAccuracy
from openrl.rewards.base_reward import BaseReward


class NLPReward(BaseReward):
    def __init__(self, env: Env, ref_model: str, intent_model: str):
        from openrl.envs.nlp.nlp_rewards import KLPenalty

        self.rew_infos = []

        reward_config = {
            "model_path": intent_model,
            "intent_coeff": 0.5,
            "auto_coeff": 0.5,
            "debug": False,
        }
        self.inner_reward_fn = [IntentAccuracy(**reward_config)]
        self.step_rew_funcs = []
        self.batch_rew_funcs = {
            "kl": KLPenalty(env.action_space, ref_model),
        }

    def get_reward(self, data: Dict[str, Any]):
        rewards = data["reward"].copy()
        infos = None

        for rew_func in self.step_rew_funcs:
            new_rew, new_info = rew_func(data)
            if infos is None:
                infos = new_info
            else:
                for i in range(len(infos)):
                    infos[i].update(new_info[i])

            rewards += new_rew

        self.rew_infos.append(infos)

        return rewards, infos

    def batch_rewards(self, buffer: NormalReplayBuffer):
        """update KL_penalty's alpha coef per episode

        Args:
            buffer (NormalReplayBuffer): Not used
        """

        infos = dict()
        if len(self.batch_rew_funcs) > 0:
            for step in range(buffer.data.actions.shape[0]):
                obs = buffer.data.all_batch_data("policy_obs", min=step, max=step + 1)
                actions = buffer.data.all_batch_data("actions", min=step, max=step + 1)
                action_log_probs = buffer.data.all_batch_data(
                    "action_log_probs", min=step, max=step + 1
                )

                data = {
                    "obs": obs,
                    "actions": actions,
                    "action_log_probs": action_log_probs,
                }
                for rew_func in self.batch_rew_funcs.values():
                    new_rew, new_info = rew_func(data)
                    for key in new_info:
                        if key not in infos:
                            infos[key] = [new_info[key]]
                        else:
                            infos[key].append(new_info[key])
                    buffer.data.rewards[step] += new_rew.reshape(
                        buffer.data.rewards[step].shape
                    )

            for key in infos:
                infos[key] = np.array(infos[key]).mean()

            if "kl" in self.batch_rew_funcs:
                self.batch_rew_funcs["kl"].update_alpha(infos["kl_div"])

        return infos
