from typing import Any, Dict, Union

import torch
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from openrl.supports.opengpu.manager import LocalGPUManager
from openrl.supports.opendata.utils.opendata_utils import data_abs_path
from openrl.envs.nlp.utils.custom_text_generation_pools import DailyDialog

class Intent():
    def __init__(self, intent_model: str, intent_coeff: float = 1.) -> None:
        super().__init__()
        
        self._intent_coeff = intent_coeff

        model_path = data_abs_path(intent_model)
        self._tokenizer = AutoTokenizer.from_pretrained(intent_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)

        if torch.cuda.is_available():
            manager = LocalGPUManager()
            manager.log_info()
            self._device = f"cuda:{manager.get_gpu()}"
        else:
            self._device = "cpu"
        print("Intent Model choose to use device:{}".format(self._device))

        self._model = self._model.to(self._device)

    def __call__(
        self,
        data: Dict[str, Any],
    ) -> Union[np.ndarray, Dict[str, Any]]:
        
        meta_infos = data['meta_infos']
        prompt_texts = data['prompt_texts']
        generated_texts = data['generated_texts']
        
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        score = (np.array(pred_labels) == np.array(target_intents)) * 1.
        
        rewards = score * self._intent_coeff
        infos = {'intent': np.mean(score)}
        
        return rewards, infos
