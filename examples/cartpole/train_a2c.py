""""""

import numpy as np
import torch

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import A2CNet as Net
from openrl.runners.common import A2CAgent as Agent


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "a2c.yaml"])

    # create environment, set environment parallelism to 9
    env = make("CartPole-v1", env_num=9)

    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # initialize the trainer
    agent = Agent(net, use_wandb=False, project_name="CartPole-v1")
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=30000)

    env.close()

    agent.save("./a2c_agent")
    return agent


def evaluation():
    # begin to test

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "a2c.yaml"])

    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = "group_human"
    render_mode = None
    env = make("CartPole-v1", render_mode=render_mode, env_num=9, asynchronous=True)

    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # initialize the trainer
    agent = Agent(
        net,
    )
    agent.load("./a2c_agent")
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False

    total_step = 0
    total_reward = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        total_step += 1
        total_reward += np.mean(r)
        if total_step % 50 == 0:
            print(f"{total_step}: reward:{np.mean(r)}")
    env.close()
    print("total step:", total_step)
    print("total reward:", total_reward)


if __name__ == "__main__":
    train()
    evaluation()
