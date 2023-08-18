# Modified from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py
import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from openrl.envs.vec_env import BaseVecEnv, SyncVectorEnv
from openrl.utils import type_aliases


def evaluate_policy(
    agent: "type_aliases.AgentActor",
    env: Union[gym.Env, BaseVecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[float, float], Tuple[List[float], List[int]]
]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param agent: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``BaseVecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from openrl.envs.wrappers.monitor import Monitor

    if not isinstance(env, BaseVecEnv):
        env = SyncVectorEnv([lambda: env])

    is_monitor_wrapped = env.env_is_wrapped(Monitor, indices=0)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            (
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. This"
                " may result in reporting modified episode lengths and rewards, if"
                " other wrappers happen to modify these. Consider wrapping environment"
                " first with ``Monitor`` wrapper."
            ),
            UserWarning,
        )

    n_envs = env.parallel_env_num
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros([n_envs, env.agent_num])
    current_lengths = np.zeros(n_envs, dtype="int")
    # get the train_env, and will set it back after evaluation
    train_env = agent.get_env()
    agent.set_env(env)
    observations, info = env.reset()
    states = None
    episode_starts = np.ones((env.parallel_env_num,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, states = agent.act(
            observations,
            deterministic=deterministic,
        )
        observations, rewards, dones, infos = env.step(actions)
        rewards = np.squeeze(rewards, axis=-1)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                all_dones = np.all(dones[i])
                done = all_dones
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if all_dones:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        assert "final_info" in info.keys(), (
                            "final_info should be in info keys",
                            info.keys(),
                        )
                        assert "episode" in info["final_info"].keys(), (
                            "episode should be in final_info keys",
                            info["final_info"].keys(),
                        )
                        if "episode" in info["final_info"].keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["final_info"]["episode"]["r"])
                            episode_lengths.append(info["final_info"]["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(copy.copy(current_rewards[i]))
                        episode_lengths.append(copy.copy(current_lengths[i]))
                        episode_counts[i] += 1

                    current_rewards[i] = 0
                    current_lengths[i] = 0

        # if render:
        #     env.render()
    # set env to train_env
    agent.set_env(train_env)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert (
            mean_reward > reward_threshold
        ), f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
