import time
import warnings
from typing import Optional

import gym
import gymnasium
import numpy as np
from envpool.python.protocol import EnvPool
from packaging import version
from stable_baselines3.common.vec_env import VecEnvWrapper as BaseWrapper
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class VecEnvWrapper(BaseWrapper):
    @property
    def agent_num(self):
        if self.is_original_envpool_env():
            return 1
        else:
            return self.env.agent_num

    def is_original_envpool_env(self):
        return not hasattr(self.venv, "agent_num`")


class VecAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        venv.num_envs = venv.spec.config.num_envs
        observation_space = venv.observation_space
        new_observation_space = gymnasium.spaces.Box(
            low=observation_space.low,
            high=observation_space.high,
            dtype=observation_space.dtype,
        )
        action_space = venv.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            new_action_space = gymnasium.spaces.Discrete(action_space.n)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            new_action_space = gymnasium.spaces.MultiDiscrete(action_space.nvec)
        elif isinstance(action_space, gym.spaces.MultiBinary):
            new_action_space = gymnasium.spaces.MultiBinary(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            new_action_space = gymnasium.spaces.Box(
                low=action_space.low,
                high=action_space.high,
                dtype=action_space.dtype,
            )
        else:
            raise NotImplementedError(f"Action space {action_space} is not supported")
        super().__init__(
            venv=venv,
            observation_space=new_observation_space,
            action_space=new_action_space,
        )

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        if is_legacy_gym:
            return self.venv.reset(), {}
        else:
            return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        if is_legacy_gym:
            obs, rewards, dones, info_dict = self.venv.step(self.actions)
        else:
            obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
            dones = terms + truncs
        rewards = rewards
        infos = []
        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[key][i]
                    for key in info_dict.keys()
                    if isinstance(info_dict[key], np.ndarray)
                }
            )
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                if is_legacy_gym:
                    obs[i] = self.venv.reset(np.array([i]))
                else:
                    obs[i] = self.venv.reset(np.array([i]))[0]
        return obs, rewards, dones, infos


class VecMonitor(VecEnvWrapper):
    def __init__(
        self,
        venv,
        filename: Optional[str] = None,
        info_keywords=(),
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapperbut you are"
                " wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics"
                " will beoverwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        self.results_writer: Optional[ResultsWriter] = None
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=info_keywords,
            )

        self.info_keywords = info_keywords
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, **kwargs) -> VecEnvObs:
        obs, info = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        rewards = np.expand_dims(rewards, 1)
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()


__all__ = ["VecAdapter", "VecMonitor"]
