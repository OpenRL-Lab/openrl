from abc import ABC, abstractclassmethod, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_utils import unwrap_model

import openrl.envs.nlp as nlp
from openrl.envs.nlp.utils.custom_text_generation_pools import DailyDialog
from openrl.envs.nlp.utils.distribution import CategoricalDistribution
from openrl.supports.opendata.utils.opendata_utils import data_abs_path
from openrl.supports.opengpu.manager import LocalGPUManager


class KLPenalty(nn.Module):
    def __init__(
        self,
        action_space,
        model_path,
        apply_model_parallel: bool = True,
    ):
        self._apply_model_parallel = apply_model_parallel

        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        config_dict = config.to_dict()
        for key in config_dict:
            if "drop" in key:
                config_dict[key] = 0.0
        config = config.from_dict(config_dict)
        self._ref_net = AutoModelForCausalLM.from_pretrained(model_path, config=config)
        self._ref_net = self._ref_net.eval()
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._ref_net.is_parallelizable:
                self._ref_net.parallelize()
            else:  # else defaults to data parallel
                self._ref_net = torch.nn.DataParallel(self._ref_net)
        self._alpha = 0.2
        self._target_kl = 0.05
        self._update_rate = 0.1
        self._action_dist = CategoricalDistribution(action_space.n)

    def update_alpha(self, kl_div):
        diff_to_target = (kl_div - self._target_kl) / self._target_kl
        e_t = np.clip(diff_to_target, -0.2, 0.2)
        self._alpha = self._alpha * (1 + self._update_rate * e_t)

    def __call__(self, data: Dict[str, Any], past_model_kwargs: Any = None) -> Any:
        actions = data["actions"]
        action_log_probs = data["action_log_probs"]
        obs = data["obs"]

        actions = torch.tensor(actions).flatten()

        input_ids = torch.tensor(obs["input_encoded_pt"]).int()
        attention_mask = torch.tensor(obs["input_attention_mask_pt"])

        self._ref_net = self._ref_net.eval()

        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }

        model_inputs = self._prepare_inputs_for_model(
            self._ref_net, input_ids, past_model_kwargs
        )

        output = self._ref_net(output_hidden_states=True, **model_inputs)
        output["past_key_values"] = None
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        action_input = actions.to(next_token_logits.device)
        ref_log_prob = dist.log_prob(action_input)

        ref_log_prob = ref_log_prob.reshape(action_log_probs.shape)
        kl_div = action_log_probs.copy() - ref_log_prob.detach().cpu().numpy()
        rew = -self._alpha * kl_div
        infos = {
            "alpha": self._alpha,
            "kl_div": kl_div.mean(),
        }
        return rew, infos

    def _prepare_inputs_for_model(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )

        if self._apply_model_parallel and unwrap_model(model).is_parallelizable:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {
                key: (
                    value.to(model.transformer.first_device)
                    if isinstance(value, torch.Tensor)
                    and hasattr(model.transformer, "first_device")
                    else value
                )
                for key, value in model_inputs.items()
            }
        return model_inputs


class BatchedRewardFunction(ABC):
    """
    Computes rewards for several instances at once
    """

    @abstractclassmethod
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        An abstract class for batched reward functions for text generation
        """
        raise NotImplementedError


class IntentAccuracy(BatchedRewardFunction):
    def __init__(
        self,
        model_path: str,
        shape: bool = True,
        intent_coeff: float = 1.0,
        auto_coeff: float = 1.0,
        debug=False,
    ) -> None:
        super().__init__()
        self.debug = debug
        if debug:
            return
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._model_path = model_path
        self._shaping_metric = None  # MeteorMetric()

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        # compute rewards for finished episodes only

        meteor_rewards = np.zeros(len(gen_texts))
        intent_rewards = np.zeros(len(gen_texts))
        if self.debug:
            reward_info = {
                "meteor_rewards": meteor_rewards,
                "intent_rewards": intent_rewards,
            }
            return meteor_rewards + intent_rewards, reward_info
        if self._metric is None:
            self._metric = IntentAccuracyDailyDialog(self._model_path)
        if self._shaping_metric is None:
            self._shaping_metric = MeteorMetric()

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    meteor_rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        if len(done_prompt_texts):
            scores = self._metric.compute(
                done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
            )["intent/accuracy"][0]
            intent_rewards[done_ixs] += self._intent_coeff * np.array(scores)

        reward_info = {
            "meteor_rewards": meteor_rewards,
            "intent_rewards": intent_rewards,
        }

        return meteor_rewards + intent_rewards, reward_info  # .tolist()


class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: Optional[str] = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        # self._metric = evaluate.load("meteor.py")
        self._metric = evaluate.load(
            str(Path(nlp.__file__).parent / "utils/metrics/meteor.py")
        )

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: Optional[str] = None,
    ):
        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class IntentAccuracyDailyDialog(BaseMetric):
    def __init__(self, model_path: str) -> None:
        super().__init__()

        model_path = data_abs_path(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)

        if torch.cuda.is_available():
            manager = LocalGPUManager()
            manager.log_info()
            self._device = f"cuda:{manager.get_gpu()}"
        else:
            self._device = "cpu"
        print("Intent Model choose to use device:{}".format(self._device))

        self._model = self._model.to(self._device)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: Optional[str] = None,
    ) -> Tuple[List[float], float]:
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

        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(
            np.int32
        )
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict
