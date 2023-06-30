"""
Used for generate offline data for GAIL.
"""

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env.wrappers.gen_data import GenDataWrapper
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


def gen_data(total_episode):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
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

    env = GenDataWrapper(env, data_save_path="data.pkl", total_episode=total_episode)
    obs, info = env.reset()
    done = False
    while not done:
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
    env.close()


if __name__ == "__main__":
    train()
    gen_data(total_episode=500)
