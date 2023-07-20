# Modified from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym

import openrl.utils.callbacks.callbacks_factory as callbacks_factory
from openrl.envs.vec_env import BaseVecEnv
from openrl.runners.common.base_agent import BaseAgent
from openrl.utils.logger import Logger


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import
    agent: "BaseAgent"
    logger: Logger

    def __init__(self, verbose: int = 0):
        super().__init__()
        # An alias for self.agent.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, BaseVecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_time_steps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, agent: "BaseAgent") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.agent = agent
        self.training_env = agent.get_env()
        self.logger = agent.logger
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_time_steps = self.agent.num_time_steps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_time_steps = self.agent.num_time_steps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass

    def set_parent(self, parent: "BaseCallback") -> None:
        """
        Set the parent of the callback.

        :param parent: The parent callback.
        """
        self.parent = parent


class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: Callback that will be called
        when an event is triggered.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.set_parent(self)

    def init_callback(self, agent: "BaseAgent") -> None:
        super().init_callback(agent)
        if self.callback is not None:
            self.callback.init_callback(self.agent)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback is not None:
            self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: List[BaseCallback], stop_logic: str = "OR"):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks
        self.stop_logic = stop_logic

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.agent)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        if self.stop_logic == "OR":
            # any callback return should_stop, then to stop
            should_stop = False
        elif self.stop_logic == "AND":
            # all callbacks return should_stop, then to stop
            should_stop = True
        else:
            raise ValueError(
                "Unknown stop logic {}, possible values are 'OR' or 'AND'".format(
                    self.stop_logic
                )
            )

        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            if self.stop_logic == "OR":
                should_stop = (not callback.on_step()) or should_stop

            elif self.stop_logic == "AND":
                should_stop = (not callback.on_step()) and should_stop

        return not should_stop

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)

    def __repr__(self):
        callback_names = []
        for callback in self.callbacks:
            callback_names.append(callback.__class__.__name__)
        return str(callback_names)

    def set_parent(self, parent: "BaseCallback") -> None:
        """
        Set the parent of the callback.

        :param parent: The parent callback.
        """
        self.parent = parent
        for callback in self.callbacks:
            callback.set_parent(parent)


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.

    :param callback:
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        callback: Callable[[Dict[str, Any], Dict[str, Any]], bool],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True


class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(
        self,
        n_steps: int,
        callbacks: Union[List[Dict[str, Any]], Dict[str, Any], BaseCallback],
        stop_logic: str = "OR",
    ):
        if isinstance(callbacks, list):
            callbacks = callbacks_factory.CallbackFactory.get_callbacks(
                callbacks, stop_logic=stop_logic
            )
        super().__init__(callbacks)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_time_steps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_time_steps
            return self._on_event()
        return True
