import io
import pathlib
from typing import List, Optional, Type, Union

import numpy as np
import torch

from openrl.runners.common.base_agent import BaseAgent, SelfAgent


class ChatAgent(BaseAgent):
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        from openrl.envs.nlp.utils.custom_text_generation_pools import DailyDialog

        self.EOU_TOKEN = DailyDialog.EOU_TOKEN

    @classmethod
    def load(
        cls: Type[SelfAgent],
        agent_path: Union[str, pathlib.Path, io.BufferedIOBase],
        tokenizer: Optional[Union[str, pathlib.Path, io.BufferedIOBase]] = None,
        disable_cuda: Optional[bool] = True,
    ) -> SelfAgent:
        if isinstance(agent_path, str):
            agent_path = pathlib.Path(agent_path)

        assert agent_path.exists(), f"{agent_path} does not exist"

        from transformers import AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(agent_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        if agent_path.is_dir():
            agent_path = agent_path / "module.pt"

        assert agent_path.exists(), f"{agent_path} does not exist"

        if not disable_cuda and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        module = torch.load(agent_path, map_location=torch.device(device))
        module.device = torch.device(device)
        module.device = torch.device(device)
        for key in module.models:
            module.models[key].tpdv = dict(
                dtype=torch.float32, device=torch.device(device)
            )

        model = module.models["model"].policy
        if device == "cpu":
            model.deparallelize()
        else:
            if model.is_parallelizable:
                model.parallelize()
            if model.config.is_encoder_decoder:
                # seq2seq LM
                device = model.encoder.first_device
            else:
                # causal LM
                device = model.transformer.first_device

        return cls(model, tokenizer, device)

    def chat(self, input: str, history: List[str]):
        from openrl.envs.nlp.utils.evaluation_utils import generate

        intput_text = self.EOU_TOKEN.join(history + [input]) + self.EOU_TOKEN
        response = generate(
            self.model,
            self.tokenizer,
            texts=intput_text,
            max_prompt_length=128,
            gen_kwargs={
                "do_sample": False,
                "top_k": 20,
                "min_length": 2,
                "max_new_tokens": 100,
                "post_processing_fn": None,
            },
            device=self.device,
        )[0]
        response = response.split(self.EOU_TOKEN)[0].strip()
        return response

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        pass


class Chat6BAgent(ChatAgent):
    @classmethod
    def load(
        cls: Type[SelfAgent],
        agent_path: Union[str, pathlib.Path, io.BufferedIOBase],
        device="cuda:0",
    ) -> SelfAgent:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(agent_path, trust_remote_code=True)
        model = (
            AutoModel.from_pretrained(agent_path, trust_remote_code=True)
            .half()
            .cuda(device)
        )
        model.eval()
        return cls(model, tokenizer, device)

    def chat(self, input: str, history: List[str]):
        new_history = np.reshape(history, (-1, 2)).tolist()
        response, _ = self.model.chat(
            self.tokenizer, input, history=new_history, do_sample=False
        )
        return response
