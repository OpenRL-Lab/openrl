from openrl.rewards.base_reward import BaseReward
from openrl.rewards.nlp_reward import NLPReward

registed_rewards = {
    "default": BaseReward,
    "NLPReward": NLPReward,
}


class RewardFactory:
    @staticmethod
    def get_reward_class(reward_class, env):
        if reward_class is None or reward_class.id is None:
            return registed_rewards["default"]()
        return registed_rewards[reward_class.id](env, **reward_class.args)

    @staticmethod
    def register(reward_name, reward_class):
        registed_rewards.update({reward_name: reward_class})
