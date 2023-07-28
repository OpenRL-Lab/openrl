""""""

from openrl.modules.common import SACNet as Net
from openrl.runners.common import SACAgent as Agent

from train_ppo import train, evaluation


if __name__ == "__main__":
    agent = train(Agent, Net, "IdentityEnvcontinuous", 10, 5000)
    evaluation(agent, "IdentityEnvcontinuous")
    # test_env()
