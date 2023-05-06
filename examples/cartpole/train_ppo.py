""""""
import numpy as np

from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # 创建 环境
    env = make("CartPole-v1", env_num=9)
    # 创建 神经网络
    net = Net(env)
    # 初始化训练器
    agent = Agent(net)
    # 开始训练
    agent.train(total_time_steps=20000)
    env.close()
    return agent


def evaluation(agent):
    # 开始测试环境
    env = make("CartPole-v1", render_mode="group_human", env_num=9, asynchronous=True)
    agent.set_env(env)
    obs, info = env.reset()
    done = False
    step = 0
    while not np.any(done):
        # 智能体根据 observation 预测下一个动作
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        print(f"{step}: reward:{np.mean(r)}")
    env.close()


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
