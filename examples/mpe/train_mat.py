""""""

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import MATNet as Net
from openrl.runners.common import MATAgent as Agent
from openrl.envs.wrappers.mat_wrapper import MATWrapper


def train():
    # 创建 环境
    env_num = 100
    env = make(
        "simple_spread",
        env_num=env_num,
        asynchronous=True,
    )
    env = MATWrapper(env)

    # 创建 神经网络
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg, device="cuda")

    # 初始化训练器
    agent = Agent(net, use_wandb=True)
    # 开始训练
    agent.train(total_time_steps=5000000)
    env.close()
    agent.save("./mat_agent/")
    return agent


def evaluation(agent):
    # render_model = "group_human"
    render_model = None
    env_num = 9
    env = make(
        "simple_spread", render_mode=render_model, env_num=env_num, asynchronous=False
    )
    env = MATWrapper(env)
    agent.load("./mat_agent/")
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


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
