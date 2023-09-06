""""""
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment, set environment parallelism to 9
    env = make("CartPole-v1", env_num=8)

    net = Net(
        env,
        cfg=cfg,
        device="cuda"
    )
    # initialize the trainer
    agent = Agent(net, use_wandb=False, project_name="CartPole-v1")
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=20000)

    env.close()
    return agent


def evaluation(agent):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = "group_human"
    render_mode = None
    env = make("CartPole-v1", render_mode=render_mode, env_num=9, asynchronous=True)
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False
    step = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
    env.close()


if __name__ == "__main__":
    agent = train()
    # evaluation(agent)
