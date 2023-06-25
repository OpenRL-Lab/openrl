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
import numpy as np

from openrl.utils.callbacks.callbacks import BaseCallback


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``StopTrainingOnMinimumReward`` callback must be used "
            "with an ``EvalCallback``"
        )
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                "Stopping training because the mean reward"
                f" {self.parent.best_mean_reward:.2f}  is above the threshold"
                f" {self.reward_threshold}"
            )
        return continue_training


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # At start set total max according to number of environments
        self._total_max_episodes = (
            self.max_episodes * self.training_env.parallel_env_num
        )

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, (
            "`dones` variable is not defined, please check your code next to"
            " `callback.on_step()`"
        )
        self.n_episodes += np.sum(self.locals["dones"]).item()

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose >= 1 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.parallel_env_num
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env"
                if self.training_env.parallel_env_num > 1
                else ""
            )

            print(
                f"Stopping training with a total of {self.num_time_steps} steps because"
                " the training agent reached"
                f" max_episodes={self.max_episodes}, by playing for"
                f" {self.n_episodes} episodes {mean_ep_str}"
            )
        return continue_training


class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    def __init__(
        self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 1
    ):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``StopTrainingOnNoModelImprovement`` callback must be used with an"
            " ``EvalCallback``"
        )

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose >= 1 and not continue_training:
            print(
                "Stopping training because there was no new best model in the last"
                f" {self.no_improvement_evals:d} evaluations"
            )

        return continue_training
