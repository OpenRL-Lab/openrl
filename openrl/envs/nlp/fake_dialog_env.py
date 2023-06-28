from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete
from gymnasium.spaces.dict import Dict as DictSpace


class FakeDialogEnv(Env):
    def __init__(
        self,
        cfg,
        max_episode_length: int = 4,
        max_prompt_length: Optional[int] = 3,
        terminate_on_eos: bool = True,  # False,
        context_start_token: Optional[int] = None,
        prompt_truncation_side: str = "left",
    ):
        self.env_name = "fake_dialog"

        self.max_steps = max_episode_length
        self._max_text_length = max_prompt_length

        self._terminate_on_eos = terminate_on_eos
        self._context_start_token = context_start_token
        self._prompt_truncation_side = prompt_truncation_side
        super().__init__()

        # set the observation and action space here
        self._vocab_size = 2

        self.observation_space = DictSpace(
            {
                "input_encoded_pt": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length + self.max_steps,),
                ),
                "input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length + self.max_steps,)
                ),
            }
        )
        self.action_space = Discrete(n=self._vocab_size)

        n = 2
        self.action_space = Discrete(n=n)

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__time_step = 0
        self.reward_function = None

    def set_reward(self, reward_fn):
        self.reward_function = reward_fn

    def step(
        self, action: List[int]
    ) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        self.__time_step += 1

        # decide if the episode is finished or not
        done = self.__time_step == self.max_steps
        reward = 0.0
        reward_info = dict()

        if done and self.reward_function:
            for reward_function in self.reward_function.values():
                inner_reward_data = {
                    "generated_texts": self.__current_obs.context_text,
                    "reference_texts": self.__current_obs.target_or_reference_texts,
                }
                reward_new, reward_info_new = reward_function(inner_reward_data)
                reward += reward_new
                reward_info.update(reward_info_new)

        # populate additional info
        info = dict()
        info.update(reward_info)

        obs = dict()
        for key in self.observation_space:
            obs[key] = np.zeros(self.observation_space[key].shape)

        return obs, reward, done, done, info

    def reset(self, seed=None, options=None) -> Dict[str, torch.tensor]:
        """
        Resets the environment and starts a new episode
        """
        # seed
        if seed is not None:
            np.random.seed(seed)

        obs = dict()
        for key in self.observation_space:
            obs[key] = np.zeros(self.observation_space[key].shape)
        self.__time_step = 0
        return obs, dict()

    def render(self):
        pass

    def close(self):
        pass
