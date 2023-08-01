import numpy as np
import torch
from examples.selfplay.tictactoe_utils.tictactoe_render import TictactoeRender

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import FlattenObservation
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.selfplay.wrappers.human_opponent_wrapper import HumanOpponentWrapper
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


def get_fake_env(env_num):
    env = make(
        "tictactoe_v3",
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[FlattenObservation],
        auto_reset=False,
    )
    return env


def get_human_env(env_num):
    env = make(
        "tictactoe_v3",
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[TictactoeRender, HumanOpponentWrapper],
        env_wrappers=[FlattenObservation],
        auto_reset=False,
    )
    return env


def human_vs_agent():
    env_num = 1
    fake_env = get_fake_env(env_num)
    env = get_human_env(env_num)
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(fake_env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(net)

    agent.load("./ppo_agent/")

    total_reward = 0.0
    ep_num = 5
    for ep_now in range(ep_num):
        agent.set_env(fake_env)
        obs, info = env.reset()

        done = False
        step = 0

        while not np.any(done):
            # predict next action based on the observation
            action, _ = agent.act(obs, info, deterministic=True)
            obs, r, done, info = env.step(action)
            step += 1

            if np.any(done):
                total_reward += np.mean(r) > 0
                print(f"{ep_now}/{ep_num}: reward: {np.mean(r)}")
    print(f"win rate: {total_reward / ep_num}")
    env.close()


if __name__ == "__main__":
    human_vs_agent()
