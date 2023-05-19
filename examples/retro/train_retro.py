""""""
import numpy as np

from examples.common.custom_registration import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # 创建环境，若需要并行多个环境，需要设置参数asynchronous为True；若需要设定关卡，可以设定state参数，该参数与具体游戏有关
    env = make("Airstriker-Genesis", state="Level1", env_num=2, asynchronous=True)
    # 创建网络
    net = Net(env, device="cuda")
    # 初始化训练器
    agent = Agent(net)
    # 开始训练
    agent.train(total_time_steps=2000)
    # 关闭环境
    env.close()
    return agent


def game_test(agent):
    # 开始测试环境
    env = make(
        "Airstriker-Genesis",
        state="Level1",
        render_mode="group_human",
        env_num=4,
        asynchronous=True,
    )
    agent.set_env(env)
    obs, info = env.reset()
    done = False
    step = 0
    while True:
        # 智能体根据 observation 预测下一个动作
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        print(f"{step}: reward:{np.mean(r)}")

        if any(done):
            break

    env.close()


if __name__ == "__main__":
    agent = train()
    game_test(agent)
