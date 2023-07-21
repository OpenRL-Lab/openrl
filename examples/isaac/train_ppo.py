""""""
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env import BaseVecEnv
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
# from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
# from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path

from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

import hydra
from omegaconf import DictConfig

# from rl_games.common import env_configurations, vecenv
# from rl_games.torch_runner import Runner

import datetime
import os
import torch
import pdb

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.utils import seeding

class Isaac2OpenRLWrapper:
    def __init__(self, env:VecEnvRLGames) -> BaseVecEnv:
        self.env = env
    
    @property
    def parallel_env_num(self) -> int:
        return self.env.num_envs

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        return self.env.action_space
    
    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        return self.env.observation_space

    def reset(self, **kwargs):
        """Reset all environments."""
        obs_dict = self.env.reset()
        return obs_dict['obs'].unsqueeze(1).cpu().numpy()

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        # pdb.set_trace()
        actions = torch.from_numpy(actions).squeeze(-1)

        obs_dict, self._rew, self._resets, self._extras = self.env.step(actions)

        obs = obs_dict['obs'].unsqueeze(1).cpu().numpy()
        rewards = self._rew.unsqueeze(-1).unsqueeze(-1).cpu().numpy()
        dones = self._resets.unsqueeze(-1).cpu().numpy().astype(bool)
        
        infos = []
        for i in range(dones.shape[0]):
            infos.append({})

        return obs, rewards, dones, infos

    def close(self, **kwargs):
        return self.env.close()

    @property
    def agent_num(self):
        return 1

    @property
    def use_monitor(self):
        return False
    
    @property
    def env_name(self):
        return 'Isaac-'+self.env._task.name
    
    def batch_rewards(self, buffer):
        return {}


@hydra.main(config_name="config", config_path="cfg")
def train_and_evaluate(cfg_isaac: DictConfig):
    '''
    cfg_isaac:
        defined in the cfg/config.yaml following hydra framework to build isaac sim environment.
        default task: CartPole
    cfg:
        defined in OpenRL framework to build the algorithm.
    '''

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment
    num_envs = 9 # set environment parallelism to 9
    cfg_isaac.num_envs = num_envs
    print(cfg_isaac)
    cfg_dict = omegaconf_to_dict(cfg_isaac)
    print_dict(cfg_dict)    
    headless = True # headless must be True when using Isaac sim docker.
    enable_viewport = "enable_cameras" in cfg_isaac.task.sim and cfg_isaac.task.sim.enable_cameras
    isaac_env = VecEnvRLGames(headless=headless, sim_device=cfg_isaac.device_id, enable_livestream=cfg_isaac.enable_livestream, enable_viewport=enable_viewport)
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
    total_re = 0.
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
        total_re+=np.mean(r)
    print(f"Total reward:{total_re}")
    env.close()


if __name__ == "__main__":
    train_and_evaluate()

