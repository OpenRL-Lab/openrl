""""""

import os
import sys

import numpy as np
import pytest

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


@pytest.fixture(
    scope="module",
    params=[
        "--episode_length 5 --use_recurrent_policy true --use_joint_action_loss true"
        " --use_valuenorm true --use_adv_normalize true"
    ],
)
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_train_mpe(config):
    env_num = 2
    env = make(
        "simple_spread",
        env_num=env_num,
        asynchronous=True,
    )
    net = Net(env, cfg=config)
    agent = Agent(net)
    agent.train(total_time_steps=30)
    agent.save("./ppo_agent/")
    agent.load("./ppo_agent/")
    agent.set_env(env)
    obs, info = env.reset(seed=0)
    step = 0
    while step < 5:
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        if np.any(done):
            break
        step += 1
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
