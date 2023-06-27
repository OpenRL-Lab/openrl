"""
Used for generate offline data for GAIL.
"""
import pickle

import numpy as np
from tqdm.rich import tqdm

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.monitor import Monitor
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

env_wrappers = [
    Monitor,
]


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment, set environment parallelism to 9
    env = make("CartPole-v1", env_num=9)

    net = Net(
        env,
        cfg=cfg,
    )
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=20000)
    agent.save("ppo_agent")
    env.close()
    return agent


def gen_data():
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = "group_human"
    render_mode = None
    env = make(
        "CartPole-v1",
        render_mode=render_mode,
        env_num=9,
        asynchronous=True,
        env_wrappers=env_wrappers,
    )
    agent = Agent(Net(env))
    agent.load("ppo_agent")
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    data = {
        "obs": [],
        "action": [],
        "reward": [],
        "done": [],
        "info": [],
    }
    obs, info = env.reset()
    data["action"].append(None)
    data["obs"].append(obs)
    data["reward"].append(None)
    data["done"].append(None)
    data["info"].append(info)

    total_episode = 5000
    current_total_episode = 0
    episode_lengths = []
    pbar = tqdm(total=total_episode)
    while current_total_episode < total_episode:
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        data["action"].append(action)
        obs, r, done, info = env.step(action)
        data["obs"].append(obs)
        data["reward"].append(r)
        data["done"].append(done)
        data["info"].append(info)

        for i in range(env.parallel_env_num):
            if np.all(done[i]):
                current_total_episode += 1
                pbar.update(1)
                assert "final_info" in info[i] and "episode" in info[i]["final_info"]
                episode_lengths.append(info[i]["final_info"]["episode"]["l"])

    pbar.refresh()
    pbar.close()

    print("collect total episode: {}".format(current_total_episode))
    average_length = np.mean(episode_lengths)
    print("average episode length: {}".format(average_length))

    data["total_episode"] = current_total_episode
    data["average_length"] = average_length
    env.close()

    pickle.dump(data, open("data.pkl", "wb"))


if __name__ == "__main__":
    # agent = train()
    gen_data()
