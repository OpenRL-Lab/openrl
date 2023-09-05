import numpy as np
import torch

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

env_name = "pybullet_drones/hover-aviary-v0"


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])

    # create environment, set environment parallelism to 64
    env_num = 20
    # env_num = 1

    env = make(
        env_name,
        env_num=env_num,
        cfg=cfg,
        asynchronous=True,
        env_wrappers=[],
        gui=False,
    )

    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # initialize the trainer
    agent = Agent(
        net,
    )
    # start training, set total number of training steps to 100000
    agent.train(total_time_steps=1000000)

    agent.save("./ppo_agent")
    env.close()
    return agent


def evaluation():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 4. Set rendering mode to group_rgb_array.

    env = make(
        env_name,
        env_num=1,
        asynchronous=False,
        env_wrappers=[],
        cfg=cfg,
        gui=False,
        record=False,
    )


    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # initialize the trainer
    agent = Agent(
        net,
    )
    agent.load("./ppo_agent")

    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        print("action:",action)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
        # if step % 50 == 0:
        #     print(f"{step}: reward:{np.mean(r)}")
    print("total step:", step)
    print("total reward:", total_reward)
    env.close()


if __name__ == "__main__":
    # train()
    evaluation()
