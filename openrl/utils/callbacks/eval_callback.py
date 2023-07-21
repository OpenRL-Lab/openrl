#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

import openrl.utils.callbacks.callbacks_factory as callbacks_factory
from openrl.envs.common import make
from openrl.envs.vec_env import BaseVecEnv, SyncVectorEnv
from openrl.envs.wrappers.monitor import Monitor
from openrl.utils.callbacks.callbacks import BaseCallback, EventCallback
from openrl.utils.evaluation import evaluate_policy

env_wrappers = [
    Monitor,
]


def _make_env(
    env: Union[str, Dict[str, Any]], render: bool, asynchronous: bool
) -> BaseVecEnv:
    if isinstance(env, str):
        env = {"id": env, "env_num": 1}
    envs = make(
        env["id"],
        env_num=env["env_num"],
        render_mode="group_human" if render else None,
        env_wrappers=env_wrappers,
        asynchronous=asynchronous,
    )
    return envs


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callbacks_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[str, Dict[str, Any], gym.Env, BaseVecEnv],
        callbacks_on_new_best: Optional[
            Union[List[Dict[str, Any]], Dict[str, Any], BaseCallback]
        ] = None,
        callbacks_after_eval: Optional[
            Union[List[Dict[str, Any]], Dict[str, Any], BaseCallback]
        ] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[Union[str, Path]] = None,
        best_model_save_path: Optional[Union[str, Path]] = None,
        deterministic: bool = True,
        render: bool = False,
        asynchronous: bool = True,
        verbose: int = 1,
        warn: bool = True,
        stop_logic: str = "OR",
        close_env_at_end: bool = True,
    ):
        if isinstance(callbacks_after_eval, list):
            callbacks_after_eval = callbacks_factory.CallbackFactory.get_callbacks(
                callbacks_after_eval, stop_logic=stop_logic
            )

        super().__init__(callbacks_after_eval, verbose=verbose)
        self.stop_logic = stop_logic
        if isinstance(callbacks_on_new_best, list):
            callbacks_on_new_best = callbacks_factory.CallbackFactory.get_callbacks(
                callbacks_on_new_best, stop_logic=stop_logic
            )

        self.callbacks_on_new_best = callbacks_on_new_best

        if self.callbacks_on_new_best is not None:
            # Give access to the parent
            self.callbacks_on_new_best.set_parent(self)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.close_env_at_end = close_env_at_end
        if isinstance(eval_env, str) or isinstance(eval_env, dict):
            eval_env = _make_env(eval_env, render, asynchronous)
        # Convert to BaseVecEnv for consistency
        if not isinstance(eval_env, BaseVecEnv):
            eval_env = SyncVectorEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_time_steps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callbacks_on_new_best is not None:
            self.callbacks_on_new_best.init_callback(self.agent)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_final_info = info.get("final_info")
            if maybe_final_info is not None:
                if isinstance(maybe_final_info, dict):
                    maybe_is_success = maybe_final_info.get("is_success")
                else:
                    maybe_is_success = maybe_final_info[0].get("is_success")
                if maybe_is_success is not None:
                    self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.agent,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_time_steps.append(self.num_time_steps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_time_steps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_time_steps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.agent.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                    with open(
                        os.path.join(self.best_model_save_path, "best_model_info.txt"),
                        "w",
                    ) as f:
                        f.write(f"best model at step: {self.num_time_steps}\n")
                        f.write(f"best model reward: {mean_reward}\n")
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callbacks_on_new_best is not None:
                    continue_training = self.callbacks_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

    def _on_training_end(self):
        if self.close_env_at_end:
            self.eval_env.close()
