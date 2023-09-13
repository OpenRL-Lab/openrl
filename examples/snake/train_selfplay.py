import numpy as np
import torch
from wrappers import ConvertObs

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


def train():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "selfplay.yaml"])

    # Create environment
    env_num = 10

    render_model = None

    # ConvertObs can only be used for snakes_1v1, if you want to train snakes_3v3, you need to write your own wrapper
    env = make(
        "snakes_1v1",
        render_mode=render_model,
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[ConvertObs],
        cfg=cfg,
    )
    # Create neural network

    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # Create agent
    agent = Agent(net)
    # Begin training
    agent.train(total_time_steps=100000)

    env.close()
    agent.save("./selfplay_agent/")
    return agent


def evaluation():
    from examples.selfplay.tictactoe_utils.tictactoe_render import TictactoeRender

    print("Evaluation...")
    env_num = 1
    env = make(
        "snakes_1v1",
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[ConvertObs],
        auto_reset=False,
    )

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(net)

    agent.load("./selfplay_agent/")
    agent.set_env(env)
    env.reset(seed=0)

    total_reward = 0.0
    ep_num = 5
    for ep_now in range(ep_num):
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
    print(f"win rate: {total_reward/ep_num}")
    env.close()
    print("Evaluation finished.")


if __name__ == "__main__":
    train()
    evaluation()
