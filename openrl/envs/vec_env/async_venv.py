"""An async vector environment."""
import multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import logger
from gymnasium.core import ActType, Env, ObsType
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import CloudpickleWrapper, clear_mpi_env_vars
from numpy.typing import NDArray

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.utils.numpy_utils import (
    concatenate,
    create_empty_array,
    iterate_action,
)
from openrl.envs.vec_env.utils.share_memory import (
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(BaseVecEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, and pipes for communication.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        shared_memory: bool = False,  # TODO True,
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[Callable] = None,
        render_mode: Optional[str] = None,
        auto_reset: bool = True,
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through
                shared variables. This can improve the efficiency if the observations are large (e.g. images).
            copy: If ``True``, then the :meth:`~AsyncVectorEnv.reset` and :meth:`~AsyncVectorEnv.step` methods
                return a copy of the observations.
            context: Context for `multiprocessing`_. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one.
                Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.
            render_mode: Set the render mode for the vector environment.
        Warnings: worker is an advanced mode option. It provides a high degree of flexibility and a high chance
            to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
            from the code for ``_worker`` (or ``_worker_shared_memory``) method, and add changes.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
            ValueError: If observation_space is a custom space (i.e. not a default space in Gym,
                such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
        """
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        dummy_env = env_fns[0]()
        if hasattr(dummy_env, "set_render_mode"):
            dummy_env.set_render_mode(None)

        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        self._agent_num = dummy_env.agent_num

        if hasattr(dummy_env, "env_name"):
            self._env_name = dummy_env.env_name
        elif "name" in self.metadata:
            self._env_name = self.metadata["name"]
        else:
            self._env_name = dummy_env.unwrapped.spec.id

        dummy_env.close()
        del dummy_env
        super().__init__(
            parallel_env_num=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
            render_mode=render_mode,
            auto_reset=auto_reset,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.observation_space,
                    n=self.parallel_env_num,
                    agent_num=self._agent_num,
                    ctx=ctx,
                )
                self.observations = read_from_shared_memory(
                    self.observation_space,
                    _obs_buffer,
                    n=self.parallel_env_num,
                    agent_num=self._agent_num,
                )

            except CustomSpaceError as e:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gymnasium observation spaces "
                    "(i.e. custom spaces inheriting from `gymnasium.Space`), and is "
                    "only compatible with default Gymnasium spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                ) from e
        else:
            _obs_buffer = None

            self.observations = create_empty_array(
                self.observation_space,
                n=self.parallel_env_num,
                agent_num=self._agent_num,
                fn=np.zeros,
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                        auto_reset,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    def _reset(
        self,
        seed: Union[int, List[int], None] = None,
        options: Optional[dict] = None,
    ):
        """Reset all parallel environments and return a batch of initial observations and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        self.reset_send(seed=seed, options=options)
        returns = self.reset_fetch()

        return returns

    def reset_send(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_fetch`.

        Args:
            seed: List of seeds for each environment
            options: The reset option

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`step_send`). This can be caused by two consecutive
                calls to :meth:`reset_send`, with no call to :meth:`reset_fetch` in between.
        """
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.parallel_env_num)]
        if isinstance(seed, int):
            seed = [seed + i * 10086 for i in range(self.parallel_env_num)]
        assert len(seed) == self.parallel_env_num

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                (
                    "Calling `reset_send` while waiting for a pending call to"
                    f" `{self._state.value}` to complete"
                ),
                self._state.value,
            )

        for pipe, single_seed in zip(self.parent_pipes, seed):
            single_kwargs = {}
            if single_seed is not None:
                single_kwargs["seed"] = single_seed
            if options is not None:
                single_kwargs["options"] = options

            pipe.send(("reset", single_kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_fetch(
        self,
        timeout: Optional[Union[int, float]] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Waits for the calls triggered by :meth:`reset_send` to finish and returns the results.

        Args:
            timeout: Number of seconds before the call to `reset_fetch` times out. If `None`, the call to `reset_fetch` never times out.
            seed: ignored
            options: ignored

        Returns:
            A tuple of batched observations and list of dictionaries

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`reset_fetch` was called without any prior call to :meth:`reset_send`.
            TimeoutError: If :meth:`reset_fetch` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_fetch` without any prior call to `reset_send`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `reset_fetch` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        infos = []

        results, info_data = zip(*results)
        for i, info in enumerate(info_data):
            infos.append(info)

        if not self.shared_memory:
            self.observations = concatenate(
                self.observation_space, results, self.observations
            )

        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def _step(self, actions: ActType):
        """Take an action for each parallel environment.

        Args:
            actions: element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        self.step_send(actions)
        return self.step_fetch()

    def step_send(self, actions: np.ndarray):
        """Send the calls to :obj:`step` to each sub-environment.

        Args:
            actions: Batch of actions. element of :attr:`~VectorEnv.action_space`

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`reset_send`). This can be caused by two consecutive
                calls to :meth:`step_send`, with no call to :meth:`step_fetch` in
                between.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                (
                    "Calling `step_send` while waiting for a pending call to"
                    f" `{self._state.value}` to complete."
                ),
                self._state.value,
            )

        actions = iterate_action(self.action_space, actions)

        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_fetch(
        self, timeout: Optional[Union[int, float]] = None
    ) -> Union[
        Tuple[Any, NDArray[Any], NDArray[Any], List[Dict[str, Any]]],
        Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], List[Dict[str, Any]]],
    ]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_fetch` times out. If ``None``, the call to :meth:`step_fetch` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`step_fetch` was called without any prior call to :meth:`step_send`.
            TimeoutError: If :meth:`step_fetch` timed out.
        """

        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_fetch` without any prior call to `step_send`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `step_fetch` has timed out after {timeout} second(s)."
            )

        observations_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        result_len = None
        successes = []

        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()

            successes.append(success)
            if success:
                if result_len is None:
                    result_len = len(result)
                if result_len == 5:
                    obs, rew, terminated, truncated, info = result
                    truncateds.append(truncated)
                elif result_len == 4:
                    obs, rew, terminated, info = result
                else:
                    raise ValueError(
                        f"Invalid number of return values from step: {result_len}"
                    )
                terminateds.append(terminated)
                observations_list.append(obs)
                rewards.append(rew)

                infos.append(info)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(
                self.observation_space,
                observations_list,
                self.observations,
            )

        assert result_len in (
            4,
            5,
        ), f"Invalid number of return values from step: {result_len}"
        if result_len == 4:
            return (
                deepcopy(self.observations) if self.copy else self.observations,
                np.array(rewards),
                np.array(terminateds, dtype=np.bool_),
                infos,
            )
        else:
            return (
                deepcopy(self.observations) if self.copy else self.observations,
                np.array(rewards),
                np.array(terminateds, dtype=np.bool_),
                np.array(truncateds, dtype=np.bool_),
                infos,
            )

    def close_extras(
        self, timeout: Optional[Union[int, float]] = None, terminate: bool = False
    ):
        """Close the environments & clean up the extra resources (processes and pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending call to"
                    f" `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_fetch")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_spaces(self):
        self._assert_is_running()
        spaces = (self.observation_space, self.action_space)
        for pipe in self.parent_pipes:
            pipe.send(("_check_spaces", spaces))
        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        same_observation_spaces, same_action_spaces = zip(*results)
        if not all(same_observation_spaces):
            raise RuntimeError(
                "Some environments have an observation space different from "
                f"`{self.observation_space}`. In order to batch observations, "
                "the observation spaces from all environments must be equal."
            )
        if not all(same_action_spaces):
            raise RuntimeError(
                "Some environments have an action space different from "
                f"`{self.action_space}`. In order to batch actions, the "
                "action spaces from all environments must be equal."
            )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to"
                " `close()`."
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.parallel_env_num - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                f"Received the following error from Worker-{index}: {exctype.__name__}:"
                f" {value}"
            )
            logger.error(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)

    def _get_images(self) -> Sequence[np.ndarray]:
        self._assert_is_running()
        if self.render_mode == "single_rgb_array":
            pipe = self.parent_pipes[0]
            pipe.send(("_call", ("render", [], {})))
            results = [pipe.recv()]
        else:
            for pipe in self.parent_pipes:
                pipe.send(("_call", ("render", [], {})))
            results = [pipe.recv() for pipe in self.parent_pipes]

        imgs, successes = zip(*results)
        self._raise_if_errors(successes)
        return imgs

    @property
    def env_name(self):
        return self._env_name

    def call_send(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_send` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                (
                    "Calling `call_send` while waiting "
                    f"for a pending call to `{self._state.value}` to complete."
                ),
                str(self._state.value),
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_fetch(self, timeout: Union[int, float, None] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `call_fetch` times out.
                If `None` (default), the call to `call_fetch` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_fetch` without any prior call to `call_send`.
            TimeoutError: The call to `call_fetch` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_fetch` without any prior call to `call_send`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_fetch` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def exec_func_send(self, func: Callable, indices, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            func: a function.
            indices: Indices of the environments to call the method on.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_send` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                (
                    "Calling `exec_func_send` while waiting "
                    f"for a pending call to `{self._state.value}` to complete."
                ),
                str(self._state.value),
            )

        for pipe in self.parent_pipes:
            pipe.send(("_func_exec", (func, indices, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def exec_func_fetch(self, timeout: Union[int, float, None] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `exec_func_fetch` times out.
                If `None` (default), the call to `exec_func_fetch` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_fetch` without any prior call to `call_send`.
            TimeoutError: The call to `call_fetch` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `exec_func_fetch` without any prior call to `exec_func_send`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_fetch` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def get_attr(self, name: str):
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: Union[List[Any], Tuple[Any], object]):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.parallel_env_num)]
        if len(values) != self.parallel_env_num:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.parallel_env_num} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                (
                    "Calling `set_attr` while waiting "
                    f"for a pending call to `{self._state.value}` to complete."
                ),
                str(self._state.value),
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)


def _worker(
    index: int,
    env_fn: callable,
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory: bool,
    error_queue: Queue,
    auto_reset: bool = True,
):
    env = env_fn()
    observation_space = env.observation_space
    action_space = env.action_space

    _subenv_auto_reset = hasattr(env, "has_auto_reset") and env.has_auto_reset
    _agent_num = env.agent_num

    def prepare_obs(observation):
        if shared_memory:
            write_to_shared_memory(
                observation_space,
                _agent_num,
                index,
                observation,
                shared_memory,
            )
            observation = None
        return observation

    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                result = env.reset(**data)

                if isinstance(result, tuple):
                    assert len(result) == 2, (
                        "The `reset` method of the environment must return either a"
                        " single observation or a tuple of (observation, info)."
                    )
                    observation, info = result
                    observation = prepare_obs(observation)
                    pipe.send(((observation, info), True))
                else:
                    observation = result
                    observation = prepare_obs(observation)
                    pipe.send(((observation,), True))

            elif command == "step":
                result = env.step(data)

                result_len = len(result)
                _need_reset = not _subenv_auto_reset

                if result_len == 4:
                    (
                        observation,
                        reward,
                        terminated,
                        info,
                    ) = result
                    need_reset = _need_reset and all(terminated)
                elif result_len == 5:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = result
                    need_reset = _need_reset and (all(terminated) or all(truncated))
                else:
                    raise NotImplementedError(
                        "Step result length can not be {}.".format(result_len)
                    )
                if need_reset and auto_reset:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info = deepcopy(info)
                    info["final_observation"] = old_observation
                    info["final_info"] = old_info

                observation = prepare_obs(observation)

                if result_len == 4:
                    pipe.send(((observation, reward, terminated, info), True))
                else:
                    pipe.send(
                        ((observation, reward, terminated, truncated, info), True)
                    )

            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_check_spaces":
                pipe.send(
                    (
                        (data[0] == observation_space, data[1] == action_space),
                        True,
                    )
                )
            elif command == "_func_exec":
                function, indices, args, kwargs = data
                if index in indices:
                    if callable(function):
                        pipe.send((function(env, *args, **kwargs), True))
                    else:
                        pipe.send((function, True))
                else:
                    pipe.send((None, True))
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))

            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
