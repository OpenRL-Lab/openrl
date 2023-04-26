from typing import Any, Dict, Optional

import torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.dict import Dict as DictSpace
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import unwrap_model

from openrl.modules.networks.utils.nlp.base_policy import (
    EvaluateActionsOutput,
    GenerationInputs,
    LMActorCriticPolicy,
    PolicyOutput,
    PolicyType,
    ValueOutput,
)
from openrl.modules.networks.utils.nlp.hf_generation_utils import (
    override_generation_routines,
    unwrap_generation_routines,
)
from openrl.modules.utils.valuenorm import ValueNorm


class CausalLMActorCriticPolicy(LMActorCriticPolicy):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        model_name: str,
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
        config: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(
            observation_space,
            action_space,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
            config,
            device,
        )
        self.load_from_dict(state_dict)

    def load_from_dict(self, state_dict: dict = None):
        if state_dict is not None:
            self._policy_model.load_state_dict(state_dict["policy_model"])
            self._value_model.load_state_dict(state_dict["value_model"])
            self._value_head.load_state_dict(state_dict["value_head"])
            self.optimizer.load_state_dict(state_dict["optimizer"])

    @property
    def policy(self):
        policy_model = self._policy_model
        policy_model.__class__ = unwrap_generation_routines(type(policy_model))
        return policy_model

    def _build_model_heads(self, model_name: str, config: str, device: str):
        if self.disable_drop_out:
            config = AutoConfig.from_pretrained(model_name)
            config_dict = config.to_dict()
            for key in config_dict:
                if "drop" in key:
                    config_dict[key] = 0.0
            config = config.from_dict(config_dict)

        self._policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config
        )

        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model)
        )

        self._value_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config
        )

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False
        )
        self.value_normalizer = (
            ValueNorm(1, device=device) if self._use_valuenorm else None
        )

        torch.multiprocessing.set_sharing_strategy("file_system")
        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
                if self._use_valuenorm:
                    self.value_normalizer.to(self.device)
            else:  # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(
                    self._value_head.to(self.device)
                )
                if self._use_valuenorm:
                    self.value_normalizer = torch.nn.DataParallel(
                        self.value_normalizer.to(self.device)
                    )

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

    def forward_policy(
        self,
        obs,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            # take attention mask only for the first step
            # for subsequent steps, update_model_kwargs will handle it
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._policy_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._policy_model(output_hidden_states=True, **model_inputs)
        output["past_key_values"] = None
        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        entropy = dist.entropy()

        # sample act
        actions_input = actions.to(next_token_logits.device)
        log_prob = dist.log_prob(actions_input)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._policy_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._policy_model
            ).config.is_encoder_decoder,
        )

        policy_outputs = PolicyOutput(
            actions=actions,
            raw_log_probs=log_prob,
            log_probs=log_prob,
            entropy=entropy,
            past_model_kwargs=past_model_kwargs,
        )

        return policy_outputs

    def forward_value(
        self,
        obs,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._value_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._value_model(output_hidden_states=True, **model_inputs)
        output["past_key_values"] = None
        # pool the hidden states ?
        last_tokens_hidden = output.hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._value_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._value_model
            ).config.is_encoder_decoder,
        )

        value_outputs = ValueOutput(values=values, past_model_kwargs=past_model_kwargs)

        return value_outputs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:
        policy_outputs = self.forward_policy(obs=obs, actions=actions)
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def get_policy_first_device(self):
        return (
            self._policy_model.transformer.first_device
            if self._apply_model_parallel
            and unwrap_model(self._policy_model).is_parallelizable
            else "cuda"
        )

    def get_inputs_for_generation(self, obs):
        gen_inputs = GenerationInputs(
            obs["input_encoded_pt"], obs["input_attention_mask_pt"]
        )
        return gen_inputs

    def get_policy_type(self):
        return PolicyType.CAUSAL

    def to(self, device: str):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)

    def get_distribution(self, obs, past_model_kwargs, detach=False):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        if past_model_kwargs is None:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        if detach:
            with torch.no_grad():
                model_inputs = self._prepare_inputs_for_model(
                    self._policy_model, input_ids, past_model_kwargs
                )

                # forward pass to transformers
                output = self._policy_model(output_hidden_states=True, **model_inputs)
        else:
            model_inputs = self._prepare_inputs_for_model(
                self._policy_model, input_ids, past_model_kwargs
            )

            # forward pass to transformers
            output = self._policy_model(output_hidden_states=True, **model_inputs)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._policy_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._policy_model
            ).config.is_encoder_decoder,
        )
        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)

        return dist, past_model_kwargs

    def predict_values(self, obs):
        return self.forward_value(obs).values
