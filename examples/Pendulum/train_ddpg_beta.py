""""""
import numpy as np

# from openrl.envs.toy_envs import make
from openrl.envs.common import make
from openrl.modules.common import DDPGNet as Net
from openrl.runners.common import DDPGAgent as Agent
from openrl.configs.config import create_config_parser


def train():
    # 添加读取配置文件的代码
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ddpg_pendulum.yaml"])

    # 创建 环境
    env = make("Pendulum-v1", env_num=5)
    # 创建 神经网络
    net = Net(env, cfg=cfg)
    # 初始化训练器
    agent = Agent(net)
    # 开始训练
    agent.train(total_time_steps=100000)
    env.close()
    return agent


def evaluation(agent):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    env = make("Pendulum-v1", render_mode="group_human", env_num=4, asynchronous=True)
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False
    step = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
    env.close()


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
