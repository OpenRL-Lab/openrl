import numpy as np
from gymnasium.wrappers import FlattenObservation

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers.base_wrapper import BaseWrapper
from openrl.envs.wrappers.extra_wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


class FrameSkip(BaseWrapper):
    def __init__(self, env, num_frames: int = 8):
        super().__init__(env)
        self.num_frames = num_frames

    def step(self, action):
        num_skips = self.num_frames
        total_reward = 0.0

        for x in range(num_skips):
            obs, rew, term, trunc, info = super().step(action)
            total_reward += rew
            if term or trunc:
                break

        return obs, total_reward, term, trunc, info


env_name = "dm_control/cartpole-balance-v0"
env_name = "dm_control/walker-walk-v0"


def train():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])

    # create environment, set environment parallelism to 9
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
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=4000000)

    env.close()
    return agent


agent = train()


def evaluation(agent):
    # begin to test
    # Create an environment for testing and set the number of environments to interact with to 9. Set rendering mode to group_human.
    render_mode = "group_human"
    render_mode = "group_rgb_array"
    env = make(
        env_name,
        render_mode=render_mode,
        env_num=4,
        asynchronous=True,
        env_wrappers=[FlattenObservation],
    )
    env = GIFWrapper(env, gif_path="./new.gif", fps=50)
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
    print("total step:", step, total_reward)
    env.close()


evaluation(agent)
