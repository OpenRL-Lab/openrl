from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete
from gymnasium.spaces.dict import Dict as DictSpace
from transformers import AutoTokenizer

from openrl.envs.nlp.utils.custom_text_generation_pools import DailyDialog
from openrl.envs.nlp.utils.observation import Observation
from openrl.envs.nlp.utils.sampler import PrioritySampler
from openrl.envs.nlp.utils.text_generation_pool import Sample


class DailyDialogEnv(Env):
    def __init__(
        self,
        cfg,
        max_episode_length: int = 20,
        priority_scale: float = 0.0,
        max_prompt_length: Optional[int] = 128,
        terminate_on_eos: bool = True,  # False,
        context_start_token: Optional[int] = None,
        prompt_truncation_side: str = "left",
    ):
        """
        A generic RL environment to generate textual sequences.
        For eg: text generation, summarization, machine translation, text simplification
        Args:
            max_episode_length (int, optional): Max steps to the model Defaults to 512.
            priority_scale (float, optional): weight for the priority sampler Defaults to 0.0.
            max_prompt_length (Optional[int], optional): maximum prompt length. Defaults to None.
            terminate_on_eos (bool, optional): whether to terminate on EOS. Defaults to False.
            context_start_token (bool, optional): start token for the context (For Encoder-Decoder models! )
            prompt_truncation_side (str): truncation side for prompt text (Defaults to "left")
        """

        self.debug = cfg.env.args["data_path"] is None

        self.env_name = "daily_dialog"
        tokenizer_name = cfg.env.args["tokenizer_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.max_steps = max_episode_length
        self._max_text_length = (
            max_prompt_length if max_prompt_length else self.tokenizer.model_max_length
        )

        self._terminate_on_eos = terminate_on_eos
        self._context_start_token = context_start_token
        self._prompt_truncation_side = prompt_truncation_side
        super().__init__()

        # set the observation and action space here
        self._vocab_size = self.tokenizer.vocab_size

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
        # see https://github.com/huggingface/transformers/issues/4875 : rounding up to nearest power of 2 for better GPU efficiency

        if "mt5" in self.tokenizer.name_or_path:
            n = 250112
            self.action_space = Discrete(n=n)
        elif "t5" in self.tokenizer.name_or_path:
            n = 32128
            self.action_space = Discrete(n=n)

        if not self.debug:
            self.sampler_for_replaying = PrioritySampler(priority_scale=priority_scale)

            samples_config = {}
            samples_config["data_path"] = cfg.env.args["data_path"]
            samples_config["context_size"] = 5
            samples_config["split"] = "train"
            samples_config["small_debug"] = False
            samples = DailyDialog.prepare(**samples_config)

            for sample, weight in samples:
                self.sampler_for_replaying.add(sample, weight)

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__time_step = None
        self.reward_function = None

    def set_reward(self, reward_fn):
        self.reward_function = reward_fn

    def step_word(self, word: str) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        action = self.tokenizer.encode(word)[1]
        return self.step(action)

    def step(
        self, action: List[int]
    ) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        self.__time_step += 1
        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self.tokenizer)

        # decide if the episode is finished or not
        done = (action == self.tokenizer.eos_token_id and self._terminate_on_eos) or (
            self.__time_step == self.max_steps
        )

        done = done or self.__current_obs.context_text.endswith(DailyDialog.EOU_TOKEN)

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

        if done:
            batch_reward_data = {
                "generated_texts": self.__current_obs.context_text,
                "prompt_texts": self.__current_obs.prompt_or_input_text,
                "meta_infos": self.__current_obs.meta_info,
            }
            reward_info.update(batch_reward_data)

        # populate additional info
        info = dict()
        info.update(reward_info)

        obs = self.__current_obs.to_dict()

        return obs, reward, done, done, info

    def reset(self, seed=None, options=None) -> Dict[str, torch.tensor]:
        """
        Resets the environment and starts a new episode
        """
        # seed
        if seed is not None:
            np.random.seed(seed)

        if self.debug:
            obs = dict()
            for key in self.observation_space:
                obs[key] = np.zeros(self.observation_space[key].shape)
            return obs, dict()

        # gets a new sample if not provided
        sample = self.sampler_for_replaying.sample(size=1)[0]

        # init the observation
        self.__current_obs = Observation.init_from_sample(
            sample,
            self.tokenizer,
            self._max_text_length,
            self.max_steps,
            self._prompt_truncation_side,
            self._context_start_token,
            sample.meta_data,
        )

        # start the time step counter
        self.__time_step = 0

        obs = self.__current_obs.to_dict()

        return obs, dict()

    def render(self):
        pass

    def close(self):
        pass

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.sampler_for_replaying.add(sample, weight)

    def get_current_info(self):
        print(
            "prompt:{},context:{}".format(
                self.__current_obs.prompt_or_input_text, self.__current_obs.context_text
            )
        )

    def get_text(self):
        return self.__current_obs.prompt_or_input_text + "".join(
            self.__current_obs.context_text.split(" ")
        )
