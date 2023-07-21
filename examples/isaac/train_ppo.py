""""""

import numpy as np

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
import hydra
from omegaconf import DictConfig

from openrl.configs.config import create_config_parser
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

from isaac2openrl import Isaac2OpenRLWrapper


@hydra.main(config_name="config", config_path="cfg")
def train_and_evaluate(cfg_isaac: DictConfig):
    """
    cfg_isaac:
        defined in the cfg/config.yaml following hydra framework to build isaac sim environment.
        default task: CartPole
    cfg:
        defined in OpenRL framework to build the algorithm.
    """

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment
    num_envs = 9  # set environment parallelism to 9
    cfg_isaac.num_envs = num_envs
    print(cfg_isaac)
    cfg_dict = omegaconf_to_dict(cfg_isaac)
    print_dict(cfg_dict)
    headless = True  # headless must be True when using Isaac sim docker.
    enable_viewport = (
        "enable_cameras" in cfg_isaac.task.sim and cfg_isaac.task.sim.enable_cameras
    )
    isaac_env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg_isaac.device_id,
        enable_livestream=cfg_isaac.enable_livestream,
        enable_viewport=enable_viewport,
    )
    task = initialize_task(cfg_dict, isaac_env)
    env = Isaac2OpenRLWrapper(isaac_env)

    net = Net(
        env,
        cfg=cfg,
    )
    # initialize the trainer
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=40000)

    # begin to test
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs = env.reset()
    done = False
    step = 0
    total_re = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
        total_re += np.mean(r)
    print(f"Total reward:{total_re}")
    env.close()


if __name__ == "__main__":
    train_and_evaluate()
