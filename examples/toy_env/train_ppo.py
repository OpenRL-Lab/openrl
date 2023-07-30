""""""
from train_and_eval import evaluation, train

from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

if __name__ == "__main__":
    agent = train(Agent, Net, "IdentityEnvcontinuous", 10, 1000)
    evaluation(agent, "IdentityEnvcontinuous")
