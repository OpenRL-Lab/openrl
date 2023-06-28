from typing import Any

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.rewards.base_reward import BaseReward

registed_rewards = {
    "default": BaseReward,
}


class RewardFactory:
    @staticmethod
    def get_reward_class(reward_class: Any, env: BaseVecEnv):
        RewardFactory.auto_register(reward_class)
        if reward_class is None or reward_class.id is None:
            return registed_rewards["default"](env)
        return registed_rewards[reward_class.id](env, **reward_class.args)

    @staticmethod
    def register(reward_name, reward_class):
        registed_rewards.update({reward_name: reward_class})

    @staticmethod
    def auto_register(reward_class: Any):
        if reward_class is None:
            return
        if reward_class.id == "NLPReward":
            from openrl.rewards.nlp_reward import NLPReward

            registed_rewards.update({"NLPReward": NLPReward})
        elif reward_class.id == "GAILReward":
            from openrl.rewards.gail_reward import GAILReward

            registed_rewards.update({"GAILReward": GAILReward})
