""""""

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.modules.networks.policy_value_network_gpt import (
    PolicyValueNetworkGPT as PolicyValueNetwork,
)
from openrl.runners.common import PPOAgent as Agent


def train():
    # create environment
    cfg_parser = create_config_parser()
    try:
        import deepspeed
        cfg_parser = deepspeed.add_config_arguments(cfg_parser)
    except:
        print("choose not to use deepspeed in the nlp task")
    cfg = cfg_parser.parse_args()

    env_num = 2
    env = make(
        "daily_dialog",
        env_num=env_num,
        asynchronous=True,
        cfg=cfg,
    )

    # create the neural network
    model_dict = {"model": PolicyValueNetwork}
    net = Net(env, device="cuda", cfg=cfg, model_dict=model_dict)

    # initialize the trainer
    agent = Agent(net, use_wandb=False)

    # start training
    agent.train(total_time_steps=100000)
    agent.save("./ppo_agent")

    env.close()
    return agent


if __name__ == "__main__":
    agent = train()
