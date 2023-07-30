from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding

from openrl.utils.type_aliases import GymStepReturn

T = TypeVar("T", int, np.ndarray)


class IdentityEnv(gym.Env, Generic[T]):
    spec = EnvSpec("IdentityEnv")

    def __init__(
        self,
        dim: Optional[int] = None,
        space: Optional[spaces.Space] = None,
        ep_length: int = 10,
    ):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param : the action and observation space. Prospacevide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in time_steps
        """
        if space is None:
            if dim is None:
                dim = 2
            space = spaces.Discrete(dim)
        else:
            assert (
                dim is None
            ), "arguments for both 'dim' and 'space' provided: at most one allowed"
        self.dim = dim
        self.observation_space = spaces.Discrete(1)
        self.action_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> T:
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state, {}

    def step(self, action: T) -> Tuple[T, float, bool, Dict[str, Any]]:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self) -> None:
        # self.state = [self.action_space.sample()]
        self.state = [self._np_random.integers(0, self.dim)]

    def _get_reward(self, action: T) -> float:
        return 1 if np.all(self.state == action) else 0

    def render(self, mode: str = "human") -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)


class IdentityEnvcontinuous(gym.Env, Generic[T]):
    spec = EnvSpec("IdentityEnvcontinuous")

    def __init__(
        self,
        dim: Optional[int] = None,
        space: Optional[spaces.Space] = None,
        ep_length: int = 4,
    ):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param : the action and observation space. Prospacevide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in time_steps
        """
        if space is None:
            if dim is None:
                dim = 2
            space = spaces.Discrete(dim)
        else:
            assert (
                dim is None
            ), "arguments for both 'dim' and 'space' provided: at most one allowed"
        self.dim = dim
        self.state_generator = space.sample
        self.observation_space = spaces.Box(low=0, high=dim, shape=(1,))
        self.action_space = spaces.Box(low=0, high=dim - 1, shape=(1,))

        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.

        # self.reset()

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> T:
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        # print("reset:", self.state)
        return self.state, {}

    def step(self, action: T) -> Tuple[T, float, bool, Dict[str, Any]]:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self) -> None:
        # self.state = [self._np_random.randint(0, self.dim - 1)]
        self.state = [self._np_random.integers(0, self.dim)]

    def _get_reward(self, action: T) -> float:
        r = 1 - np.abs(self.state - np.clip(action, a_min=0, a_max=self.dim - 1))
        return r

    def render(self, mode: str = "human") -> None:
        pass


# Not Work Yet
class IdentityEnvBox(IdentityEnv[np.ndarray]):
    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 0.05,
        ep_length: int = 100,
    ):
        """
        Identity environment for testing purposes

        :param low: the lower bound of the box dim
        :param high: the upper bound of the box dim
        :param eps: the epsilon bound for correct value
        :param ep_length: the length of each episode in timesteps
        """
        space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        super().__init__(ep_length=ep_length, space=space)
        self.eps = eps

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _get_reward(self, action: np.ndarray) -> float:
        return (
            1.0 if (self.state - self.eps) <= action <= (self.state + self.eps) else 0.0
        )


# Not Work Yet
class IdentityEnvMultiDiscrete(IdentityEnv[np.ndarray]):
    def __init__(self, dim: int = 1, ep_length: int = 100) -> None:
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = spaces.MultiDiscrete([dim, dim])
        super().__init__(ep_length=ep_length, space=space)


# Not Work Yet
class IdentityEnvMultiBinary(IdentityEnv[np.ndarray]):
    def __init__(self, dim: int = 1, ep_length: int = 100) -> None:
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = spaces.MultiBinary(dim)
        super().__init__(ep_length=ep_length, space=space)


# Not Work Yet
class FakeImageEnv(gym.Env):
    """
    Fake image environment for testing purposes, it mimics Atari games.

    :param action_dim: Number of discrete actions
    :param screen_height: Height of the image
    :param screen_width: Width of the image
    :param n_channels: Number of color channels
    :param discrete: Create discrete action space instead of continuous
    :param channel_first: Put channels on first axis instead of last
    """

    def __init__(
        self,
        action_dim: int = 6,
        screen_height: int = 84,
        screen_width: int = 84,
        n_channels: int = 1,
        discrete: bool = True,
        channel_first: bool = False,
    ) -> None:
        self.observation_shape = (screen_height, screen_width, n_channels)
        if channel_first:
            self.observation_shape = (n_channels, screen_height, screen_width)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )
        if discrete:
            self.action_space = spaces.Discrete(action_dim)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.ep_length = 10
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self.observation_space.sample()

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        reward = 0.0
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, done, {}

    def render(self, mode: str = "human") -> None:
        pass
