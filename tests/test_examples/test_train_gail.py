""""""

import os
import sys

import pytest

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env.wrappers.gen_data import GenDataWrapper
from openrl.envs.wrappers.extra_wrappers import ZeroRewardWrapper
from openrl.envs.wrappers.monitor import Monitor
from openrl.modules.common import GAILNet as Net
from openrl.modules.common import PPONet
from openrl.runners.common import GAILAgent as Agent
from openrl.runners.common import PPOAgent


@pytest.fixture(scope="function")
def gen_data(tmpdir):
    tmp_data_path = os.path.join(tmpdir, "data.pkl")
    env_wrappers = [
        Monitor,
    ]
    print("generate data....")
    env = make(
        "CartPole-v1",
        env_num=2,
        asynchronous=True,
        env_wrappers=env_wrappers,
    )
    agent = PPOAgent(PPONet(env))
    env = GenDataWrapper(env, data_save_path=tmp_data_path, total_episode=5)
    obs, info = env.reset()
    done = False
    while not done:
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
    env.close()
    print("generate data done!")
    return tmp_data_path


@pytest.fixture(
    scope="function", params=[" --gail_use_action false", " --gail_use_action true"]
)
def config(request, gen_data):
    input_str = (
        "--episode_length 5 --use_recurrent_policy true --use_joint_action_loss true"
        " --use_valuenorm true --use_adv_normalize true --reward_class.id GAILReward"
    )
    input_str += request.param
    input_str += " --expert_data " + gen_data
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(input_str.split())
    return cfg


@pytest.mark.unittest
def test_train_gail(config):
    env = make("CartPole-v1", env_num=2, cfg=config, env_wrappers=[ZeroRewardWrapper])

    net = Net(
        env,
        cfg=config,
    )
    # initialize the trainer
    agent = Agent(net)
    agent.train(total_time_steps=200)
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
