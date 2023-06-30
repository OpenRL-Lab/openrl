""""""
import numpy as np

from openrl.envs.common import make
from openrl.modules.common import DDPGNet as Net
from openrl.runners.common import DDPGAgent as Agent
from openrl.configs.config import create_config_parser


def train():
    # 添加读取配置文件的代码
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ddpg_pendulum.yaml"])

    # 创建 环境
    # env = make("CartPole-v1")
    env = make("Pendulum-v1")
    # 创建 神经网络
    net = Net(env, cfg=cfg)
    # 初始化训练器
    agent = Agent(net)
    # 开始训练
    agent.train(total_time_steps=20000)
    env.close()
    return agent

if __name__=="__main__":
    agent = train()