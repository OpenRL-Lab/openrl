from typing import Any, Dict, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from openrl.envs.nlp.utils.custom_text_generation_pools import DailyDialog
from openrl.supports.opendata.utils.opendata_utils import data_abs_path
from openrl.supports.opengpu.manager import LocalGPUManager


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 8,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


class Intent:
    def __init__(self, intent_model: str, intent_coeff: float = 1.0) -> None:
        super().__init__()

        self._intent_coeff = intent_coeff
        self.use_deepspeed = True

        model_path = data_abs_path(intent_model)
        self._tokenizer = AutoTokenizer.from_pretrained(intent_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)

        if self.use_deepspeed:
            import deepspeed
            self._model = self._model.to('cuda')
            ds_config = get_eval_ds_config(offload=True, stage=0)
            self._model, *_ = deepspeed.initialize(model=self._model, config=ds_config)
            self._device = "cuda"
        else:
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
        meta_infos = data["meta_infos"]
        prompt_texts = data["prompt_texts"]
        generated_texts = data["generated_texts"]

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

        score = (np.array(pred_labels) == np.array(target_intents)) * 1.0

        rewards = score * self._intent_coeff
        infos = {"intent": np.mean(score)}

        return rewards, infos
