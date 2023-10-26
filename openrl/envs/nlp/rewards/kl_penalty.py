from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import unwrap_model

from openrl.envs.nlp.utils.distribution import CategoricalDistribution


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
    }
    return {
        "train_batch_size": 28,  #
        "train_micro_batch_size_per_gpu": 7,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {"enabled": True},
    }


class KLPenalty(nn.Module):
    def __init__(
        self,
        action_space: gym.Space,
        ref_model: str,
        apply_model_parallel: bool = True,
        use_deepspeed: bool = True,
    ):
        super().__init__()
        self.use_deepspeed = use_deepspeed
        self.use_fp16 = True

        # reference model
        self._apply_model_parallel = apply_model_parallel
        if ref_model == "builtin_ref":
            from transformers import GPT2Config, GPT2LMHeadModel

            config = GPT2Config()
            self._ref_net = GPT2LMHeadModel(config)
        else:
            self._ref_net = AutoModelForCausalLM.from_pretrained(ref_model)
        self._ref_net = self._ref_net.eval()
        if self.use_deepspeed:
            import deepspeed

            ds_config = get_eval_ds_config(offload=True, stage=0)
            self._ref_engine, *_ = deepspeed.initialize(model=self, config=ds_config)
        elif torch.cuda.is_available():
            if self._apply_model_parallel and self._ref_net.is_parallelizable:
                self._ref_net.parallelize()
            else:  # else defaults to data parallel
                self._ref_net = torch.nn.DataParallel(self._ref_net)

        # alpha adjustment
        self._alpha = 0.2
        self._target_kl = 0.05
        self._update_rate = 0.1
        self._clip_coef = 0.2
        self._action_dist = CategoricalDistribution(action_space.n)

    def update_alpha(self, kl_div: float) -> None:
        diff_to_target = (kl_div - self._target_kl) / self._target_kl
        e_t = np.clip(diff_to_target, -self._clip_coef, self._clip_coef)
        self._alpha = self._alpha * (1 + self._update_rate * e_t)

    def __call__(
        self, data: Dict[str, Any], past_model_kwargs: Optional[Any] = None
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        step = data["step"]
        obs = data["buffer"].data.get_batch_data("policy_obs", step)
        actions = data["actions"]
        action_log_probs = data["action_log_probs"]
        input_ids = torch.tensor(obs["input_encoded_pt"]).int()
        attention_mask = torch.tensor(obs["input_attention_mask_pt"])

        actions = torch.tensor(actions).flatten()
        input_ids = torch.squeeze(input_ids, dim=1)
        attention_mask = torch.squeeze(attention_mask, dim=1)

        self._ref_net = self._ref_net.eval()

        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._ref_net, input_ids, past_model_kwargs
        )

        if self.use_fp16:
            for key in ["input_ids", "position_ids"]:
                model_inputs[key] = model_inputs[key].half().int()
            for key in ["attention_mask"]:
                model_inputs[key] = model_inputs[key].half()

        with torch.no_grad():
            output = self._ref_net(output_hidden_states=True, **model_inputs)
            output["past_key_values"] = None
            next_token_logits = output.logits[:, -1, :]
            dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
            action_input = actions.to(next_token_logits.device)
            ref_log_prob = dist.log_prob(action_input)

        ref_log_prob = ref_log_prob.reshape(action_log_probs.shape)
        kl_div = action_log_probs.copy() - ref_log_prob.detach().cpu().numpy()
        rew = -self._alpha * kl_div
        infos = []
        for kl in kl_div:
            infos.append(
                {
                    "alpha": self._alpha,
                    "kl_div": kl.mean(),
                }
            )
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

        if self.use_deepspeed:
            model_inputs = {
                key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                for key, value in model_inputs.items()
            }

        return model_inputs
