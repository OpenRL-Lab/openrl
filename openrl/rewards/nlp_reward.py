from typing import Any, Dict, List, Union

import numpy as np
from gymnasium import Env

from openrl.envs.nlp.rewards.intent import Intent
from openrl.envs.nlp.rewards.kl_penalty import KLPenalty
from openrl.envs.nlp.rewards.meteor import Meteor
from openrl.rewards.base_reward import BaseReward


class NLPReward(BaseReward):
    def __init__(self, env: Env, ref_model: str, intent_model: str):
        self.rew_infos = []
        self.env_infos = []

        meteor_config = {
            "meteor_coeff": 0.5,
        }
        self.inner_reward_fn = {
            "meteor": Meteor(**meteor_config),
        }

        kl_config = {
            "action_space": env.action_space,
            "ref_model": ref_model,
        }
        self.step_rew_funcs = {
            "kl_pen": KLPenalty(**kl_config),
        }

        intent_config = {
            "intent_model": intent_model,
            "intent_coeff": 0.5,
        }
        self.batch_rew_funcs = {
            "intent_acc": Intent(**intent_config),
        }

    def step_reward(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        # step reward
        infos = []
        rewards = data["rewards"].copy()

        for rew_func in self.step_rew_funcs.values():
            new_rew, new_info = rew_func(data)
            if len(infos) == 0:
                infos = new_info
            else:
                for i in range(len(infos)):
                    infos[i].update(new_info[i])
            rewards += new_rew

        # collect data for alpha adjustment
        self.rew_infos.append(infos)

        # collect data for batch reward
        self.env_infos.append(data["infos"])

        return rewards, infos

    def batch_rewards(self, buffer) -> Dict[str, Any]:
        """
        calculate batch rewards and update KL_penalty's alpha coeff here.

        Args:
            buffer (): buffer.data.rewards is updated here
        """

        # collect batch data
        done_idx = []
        data = {"prompt_texts": [], "generated_texts": [], "meta_infos": []}

        for step_infos in self.env_infos:
            for env_info in step_infos:
                done = "final_info" in env_info
                done_idx.append(done)
                if done:
                    data["prompt_texts"].append(env_info["final_info"]["prompt_texts"])
                    data["generated_texts"].append(
                        env_info["final_info"]["generated_texts"]
                    )
                    data["meta_infos"].append(env_info["final_info"]["meta_infos"])
        done_idx = np.array(done_idx)

        # get batch reward
        infos = dict()
        rewards = np.zeros_like(buffer.data.rewards).flatten()
        for rew_func in self.batch_rew_funcs.values():
            new_rew, new_info = rew_func(data)
            if len(infos) == 0:
                infos = new_info
            else:
                infos.update(new_info)
            rewards[done_idx] += new_rew
        rewards = rewards.reshape(buffer.data.rewards.shape)
        buffer.data.rewards += rewards

        # update alpha
        kl = []
        for step_infos in self.rew_infos:
            for env_info in step_infos:
                kl.append(env_info["kl_div"])
        kl_div = np.array(kl).mean()
        self.step_rew_funcs["kl_pen"].update_alpha(kl_div)

        # reset vec infos
        self.rew_infos = []
        self.env_infos = []

        return infos
