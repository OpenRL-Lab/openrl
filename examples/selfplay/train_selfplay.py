import numpy as np
import torch

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import FlattenObservation
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


def train():
    # 创建 环境
    env_num = 10
    render_model = None
    env = make(
        "tictactoe_v3",
        render_mode=render_model,
        env_num=env_num,
        asynchronous=False,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[FlattenObservation],
    )
    # 创建 神经网络
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # 初始化训练器
    agent = Agent(net)
    # 开始训练
    agent.train(total_time_steps=300000)
    env.close()
    agent.save("./ppo_agent/")
    return agent


def evaluation(agent):
    render_model = "group_human"
    render_model = None
    env_num = 9
    env = make(
        "tictactoe_v3",
        render_mode=render_model,
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[FlattenObservation],
    )
    agent.load("./ppo_agent/")
    agent.set_env(env)
    obs, info = env.reset(seed=0)
    done = False
    step = 0
    total_reward = 0
    while not np.any(done):
        # 智能体根据 observation 预测下一个动作
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
    print(f"total_reward: {total_reward}")
    env.close()


def test_env():
    env_num = 1
    render_model = None
    render_model = "human"
    env = make(
        "tictactoe_v3",
        render_mode=render_model,
        env_num=env_num,
        asynchronous=False,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[FlattenObservation],
    )

    obs, info = env.reset(seed=1)
    done = False
    step_num = 0
    while not done:
        action = env.random_action(info)

        obs, done, r, info = env.step(action)

        done = np.any(done)
        step_num += 1
        if done:
            print(
                "step:"
                f" {step_num},{[env_info['final_observation'] for env_info in info]}"
            )
        else:
            print(f"step: {step_num},{obs}")


if __name__ == "__main__":
    agent = train()
    # evaluation(agent)
    # test_env()
