from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding

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
        self.metadata.update({"name": IdentityEnv})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> T:
        if seed is not None:
            self.seed(seed)
        if self._np_random is None:
            self.seed(0)
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
        assert self.dim is not None
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
