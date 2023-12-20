""""""

import os
import sys

import numpy as np
import pytest

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.atari_wrappers import (
    ClipRewardEnv,
    FireResetEnv,
    NoopResetEnv,
    WarpFrame,
)
from openrl.envs.wrappers.image_wrappers import TransposeImage
from openrl.envs.wrappers.monitor import Monitor
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

env_wrappers = [
    Monitor,
    NoopResetEnv,
    FireResetEnv,
    WarpFrame,
    ClipRewardEnv,
    TransposeImage,
]


@pytest.fixture(
    scope="module",
    params=[
        "--episode_length 5 --use_recurrent_policy false --vec_info_class.id"
        " EPS_RewardInfo --use_valuenorm true --use_adv_normalize true"
        " --use_share_model True --entropy_coef 0.01"
    ],
)
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_train_atari(config):
    env_num = 2
    env = make(
        "ALE/Pong-v5",
        env_num=env_num,
        cfg=config,
        asynchronous=True,
        env_wrappers=env_wrappers,
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
