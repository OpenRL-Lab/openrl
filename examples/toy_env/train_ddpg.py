""""""

from train_ppo import evaluation, train

from openrl.modules.common import DDPGNet as Net
from openrl.runners.common import DDPGAgent as Agent

if __name__ == "__main__":
    agent = train(Agent, Net, "IdentityEnvcontinuous", 10, 20000)
    evaluation(agent, "IdentityEnvcontinuous")
