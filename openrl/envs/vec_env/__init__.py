from typing import Optional, Type, Union

from gymnasium import Env as GymEnv

from openrl.envs.vec_env.async_venv import AsyncVectorEnv
from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.sync_venv import SyncVectorEnv
from openrl.envs.vec_env.wrappers.base_wrapper import VecEnvWrapper
from openrl.envs.vec_env.wrappers.reward_wrapper import RewardWrapper
from openrl.envs.vec_env.wrappers.vec_monitor_wrapper import VecMonitorWrapper

__all__ = [
    "BaseVecEnv",
    "SyncVectorEnv",
    "AsyncVectorEnv",
    "VecMonitorWrapper",
    "RewardWrapper",
]


def unwrap_vec_wrapper(
    env: Union[GymEnv, BaseVecEnv], vec_wrapper_class: Type[VecEnvWrapper]
) -> Optional[VecEnvWrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env:
    :param vec_wrapper_class:
    :return:
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


def is_vecenv_wrapped(
    env: Union[GymEnv, BaseVecEnv], vec_wrapper_class: Type[VecEnvWrapper]
) -> bool:
    """
    Check if an environment is already wrapped by a given ``VecEnvWrapper``.

    :param env:
    :param vec_wrapper_class:
    :return:
    """
    return unwrap_vec_wrapper(env, vec_wrapper_class) is not None
