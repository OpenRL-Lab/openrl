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
import shutil
import sys

import pytest

from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.utils.callbacks.callbacks import CallbackList, EveryNTimesteps
from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
from openrl.utils.callbacks.eval_callback import EvalCallback
from openrl.utils.callbacks.processbar_callback import ProgressBarCallback
from openrl.utils.callbacks.stop_callback import (
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)


@pytest.fixture(
    scope="module",
    params=["--seed 0"],
)
def config(request):
    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(request.param.split())
    return cfg


@pytest.mark.unittest
def test_callbacks(tmp_path, config):
    log_folder = tmp_path / "logs/callbacks/"

    env = make("CartPole-v1", env_num=3)
    agent = Agent(Net(env, cfg=config))

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_folder)
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)
    # Stop training if there is no model improvement after 2 evaluations
    callback_no_model_improvement = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=2, min_evals=1, verbose=1
    )
    eval_callback = EvalCallback(
        {"id": "CartPole-v1", "env_num": 2},
        callbacks_on_new_best=callback_on_best,
        callbacks_after_eval=callback_no_model_improvement,
        best_model_save_path=log_folder,
        log_path=log_folder,
        eval_freq=100,
        warn=False,
        close_env_at_end=False,
    )

    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(
        save_freq=1, save_path=log_folder, name_prefix="event"
    )
    event_callback = EveryNTimesteps(n_steps=500, callbacks=checkpoint_on_event)

    # Stop training if max number of episodes is reached
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100, verbose=1)

    callback = CallbackList(
        [checkpoint_callback, eval_callback, event_callback, callback_max_episodes]
    )

    agent.train(total_time_steps=1000, callback=callback)

    # Check access to local variables

    assert agent._env.observation_space.contains(callback.locals["obs"][0][0])
    # Check that the child callback was called
    assert checkpoint_callback.locals["obs"] is callback.locals["obs"]
    assert event_callback.locals["obs"] is callback.locals["obs"]
    assert checkpoint_on_event.locals["obs"] is callback.locals["obs"]
    # Check that internal callback counters match models' counters
    assert event_callback.num_time_steps == agent.num_time_steps
    assert event_callback.n_calls * agent.env_num == agent.num_time_steps

    agent.train(1000, callback=None)
    # Use progress bar
    pb_callback = ProgressBarCallback()
    agent.train(1000, callback=[checkpoint_callback, eval_callback, pb_callback])
    # Automatic wrapping, old way of doing callbacks
    agent.train(1000, callback=lambda _locals, _globals: True)

    env.close()

    max_episodes = 1
    n_envs = 2
    # CartPole-v1 has a timelimit of 200 timesteps
    max_episode_length = 200
    env = make("Pendulum-v1", env_num=n_envs)
    agent = Agent(Net(env, cfg=config))

    callback_max_episodes = StopTrainingOnMaxEpisodes(
        max_episodes=max_episodes, verbose=1
    )
    callback = CallbackList([callback_max_episodes])
    agent.train(1000, callback=callback)

    # Check that the actual number of episodes and timesteps per env matches the expected one
    episodes_per_env = callback_max_episodes.n_episodes // n_envs
    assert episodes_per_env == max_episodes
    time_steps_per_env = agent.num_time_steps // n_envs
    assert time_steps_per_env == max_episode_length

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)


@pytest.mark.unittest
def test_eval_callback_vec_env(config):
    # tests that eval callback does not crash when given a vector
    n_eval_envs = 3
    train_env = make("IdentityEnv", env_num=1)

    eval_env = make("IdentityEnv", env_num=n_eval_envs)
    train_env.reset(seed=0)
    eval_env.reset(seed=0)
    agent = Agent(Net(train_env, cfg=config))

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=100,
        warn=False,
    )
    agent.train(500, callback=eval_callback)
    assert eval_callback.last_mean_reward == 10.0


@pytest.mark.unittest
def test_eval_success_logging(tmp_path, config):
    n_bits = 2
    n_envs = 2
    env = make("BitFlippingEnv", env_num=1, n_bits=n_bits)
    eval_env = make("BitFlippingEnv", env_num=n_envs, n_bits=n_bits)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=250,
        log_path=tmp_path,
        warn=False,
    )
    env.reset(seed=0)
    eval_env.reset(seed=0)

    agent = Agent(Net(env, cfg=config))
    agent.train(500, callback=eval_callback)
    assert len(eval_callback._is_success_buffer) > 0


@pytest.mark.unittest
def test_checkpoint_additional_info(tmp_path, config):
    log_folder = tmp_path / "logs/callbacks/"

    env = make("CartPole-v1", env_num=1)
    agent = Agent(Net(env, cfg=config))

    checkpoint_callback = CheckpointCallback(
        save_freq=200,
        save_path=log_folder,
        verbose=2,
    )

    agent.train(200, callback=checkpoint_callback)

    assert os.path.exists(log_folder / "rl_model_200_steps")


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
