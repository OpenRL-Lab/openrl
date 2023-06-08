""""""

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.mat_wrapper import MATWrapper
from openrl.modules.common import VDNNet as Net
from openrl.runners.common import VDNAgent as Agent


def train():
    # create environment
    env_num = 100
    env = make(
        "simple_spread",
        env_num=env_num,
        asynchronous=True,
    )
    env = MATWrapper(env)

    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg, device="cuda")

    # initialize the trainer
    agent = Agent(net, use_wandb=True)
    # start training
    agent.train(total_time_steps=5000000)
    env.close()
    agent.save("./vdn_agent/")
    return agent


def evaluation(agent):
    # render_model = "group_human"
    render_model = None
    env_num = 9
    env = make(
        "simple_spread", render_mode=render_model, env_num=env_num, asynchronous=False
    )
    env = MATWrapper(env)
    agent.load("./vdn_agent/")
    agent.set_env(env)
    obs, info = env.reset(seed=0)
    done = False
    step = 0
    total_reward = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
    print(f"total_reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
