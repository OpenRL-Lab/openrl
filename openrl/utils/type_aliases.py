# Modifed from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/type_aliases.py

"""Common aliases for type hints"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import gym
import numpy as np
import torch as th

from openrl.envs import vec_env
from openrl.utils.callbacks import callbacks

GymEnv = Union[gym.Env, vec_env.BaseVecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Union[
    Tuple[GymObs, float, bool, Dict], Tuple[GymObs, float, bool, bool, Dict]
]
TensorDict = Dict[Union[str, int], th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[
    None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback
]


class AgentActor(Protocol):
    def act(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param deterministic: Whether to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
