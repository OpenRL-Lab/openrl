""""""
import copy

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.envs.wrappers.extra_wrappers import AddStep

env_wrappers = [AddStep]


def train(Agent, Net, env_name, env_num, total_time_steps):
    # create environment, set environment parallelism to 9
    env = make(env_name, env_num=env_num, env_wrappers=env_wrappers)
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "train.yaml"])
    net = Net(
        env,
        cfg=cfg,
    )
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=total_time_steps)
    env.close()
    return agent


def evaluation(agent, env_name):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    env = make(env_name, env_num=2, env_wrappers=env_wrappers, asynchronous=False)
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    obs = np.array([[[0, 0]], [[1, 0]]])
    action, _ = agent.act(obs, deterministic=True)
    print(obs[..., 0].flatten(), action.flatten())
    return
    done = False
    step = 0
    total_reward = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        pre_obs = copy.copy(obs)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
        print(
            f"{step}: action: {action.flatten()}, obs:{pre_obs.flatten()},reward:{np.mean(r)}"
        )
    env.close()
    print("total reward:", total_reward)


def test_env():
    env = make("IdentityEnv", env_num=1, asynchronous=False)
    obs, info = env.reset()
    print(obs)
    done = False
    step = 0
    while not np.any(done):
        action = env.random_action()
        pre_obs = copy.copy(obs)
        obs, r, done, info = env.step(action)
        step += 1
        # print(f"{step}: action: {action}, obs:{pre_obs},reward:{np.mean(r)}")
    env.close()


if __name__ == "__main__":
    agent = train(Agent, Net, "IdentityEnvcontinuous", 10, 2000)
    evaluation(agent, "IdentityEnvcontinuous")
    # test_env()
