import numpy as np
from gymnasium.wrappers import FlattenObservation

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.base_wrapper import BaseWrapper
from openrl.envs.wrappers.extra_wrappers import FrameSkip, GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

env_name = "dm_control/cartpole-balance-v0"
# env_name = "dm_control/walker-walk-v0"


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])

    # create environment, set environment parallelism to 64
    env = make(
        env_name,
        env_num=64,
        cfg=cfg,
        asynchronous=True,
        env_wrappers=[FrameSkip, FlattenObservation],
    )

    net = Net(env, cfg=cfg, device="cuda")
    # initialize the trainer
    agent = Agent(
        net,
    )
    # start training, set total number of training steps to 100000
    agent.train(total_time_steps=100000)
    agent.save("./ppo_agent")
    env.close()
    return agent


def evaluation():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 4. Set rendering mode to group_rgb_array.
    render_mode = "group_rgb_array"
    env = make(
        env_name,
        render_mode=render_mode,
        env_num=4,
        asynchronous=True,
        env_wrappers=[FrameSkip, FlattenObservation],
        cfg=cfg,
    )
    # Wrap the environment with GIFWrapper to record the GIF, and set the frame rate to 5.
    env = GIFWrapper(env, gif_path="./new.gif", fps=5)

    net = Net(env, cfg=cfg, device="cuda")
    # initialize the trainer
    agent = Agent(
        net,
    )
    agent.load("./ppo_agent")

    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    while not np.any(done):
        if step > 500:
            break
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
    print("total step:", step, "total reward:", total_reward)
    env.close()


if __name__ == "__main__":
    train()
    evaluation()
