""""""

from train_ppo import evaluation, train

from openrl.modules.common import DQNNet as Net
from openrl.runners.common import DQNAgent as Agent

if __name__ == "__main__":
    agent = train(Agent, Net, "IdentityEnv", 9, 20000)
    evaluation(agent, "IdentityEnv")
