""""""
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import SACNet as Net
from openrl.runners.common import SACAgent as Agent


def train():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment, set environment parallelism
    env = make("InvertedPendulum-v4", env_num=9)

    # create the neural network
    net = Net(env, cfg=cfg)
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps
    agent.train(total_time_steps=200000)

    env.close()
    return agent


def evaluation(agent):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = None
    render_mode = "group_human"
    env = make(
        "InvertedPendulum-v4", render_mode=render_mode, env_num=9, asynchronous=True
    )
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False
    step = 0
    totoal_reward = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action = agent.act(obs, sample=False)  # sample=False in evaluation
        obs, r, done, info = env.step(action)
        step += 1
        totoal_reward += np.mean(r)
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
    print(f"total reward: {totoal_reward}")
    env.close()


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
