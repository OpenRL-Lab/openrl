""""""
from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.modules.networks.policy_value_network_gpt import (
    PolicyValueNetworkGPT as PolicyValueNetwork,
)
from openrl.runners.common import PPOAgent as Agent


def train():
    # 创建 环境
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    env_num = 10
    env = make(
        "daily_dialog",
        env_num=env_num,
        asynchronous=True,
        cfg=cfg,
    )

    # 创建 神经网络
    model_dict = {"model": PolicyValueNetwork}
    net = Net(env, device="cuda", cfg=cfg, model_dict=model_dict)

    # 初始化训练器
    agent = Agent(net, use_wandb=True)

    # 开始训练
    agent.train(total_time_steps=100000)
    agent.save("./ppo_agent")

    env.close()
    return agent


if __name__ == "__main__":
    agent = train()
