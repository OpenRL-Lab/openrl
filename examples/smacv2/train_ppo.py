""""""

import numpy as np
from custom_vecinfo import SMACInfo
from smac_env import make_smac_envs

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env.vec_info import VecInfoFactory
from openrl.envs.wrappers.monitor import Monitor
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

VecInfoFactory.register("SMACInfo", SMACInfo)

env_wrappers = [
    Monitor,
]


def train():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment
    env_num = 10
    env = make(
        "2s_vs_1sc",
        env_num=env_num,
        asynchronous=False,
        cfg=cfg,
        make_custom_envs=make_smac_envs,
        env_wrappers=env_wrappers,
    )

    # create the neural network

    net = Net(env, cfg=cfg, device="cuda")

    # initialize the trainer
    agent = Agent(net, use_wandb=False, project_name="SMAC")
    # start training, set total number of training steps to 5000000
    agent.train(total_time_steps=10000000)
    # agent.train(total_time_steps=2000)
    env.close()
    print("Saving agent to ./ppo_agent/")
    agent.save("./ppo_agent/")

    return agent


def evaluation(agent):
    env_num = 2
    env = make(
        "2s_vs_1sc",
        env_num=env_num,
        make_custom_envs=make_smac_envs,
    )
    # agent.load("./ppo_agent/")
    agent.set_env(env)
    obs, info = env.reset(seed=0)
    done = False
    step = 0
    total_reward = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, info=info, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
        print(f"step:{step}, total_reward: {total_reward}")
    print(f"total_reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    from absl import flags

    FLAGS = flags.FLAGS
    FLAGS([""])

    agent = train()
    evaluation(agent)
