from copy import deepcopy
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedModel


def generate(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str] = None,
    max_prompt_length: int = None,
    gen_kwargs: Dict[str, Any] = {},
    device: Optional[str] = None,
):
    # switch to eval
    model.eval()

    encodings = tokenizer(
        texts,
        padding="max_length",
        max_length=max_prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
    )
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    if (
        gen_kwargs.get("min_length", None) is not None
        and not model.config.is_encoder_decoder
    ):
        gen_kwargs_ = deepcopy(gen_kwargs)
        gen_kwargs_["min_length"] = input_ids.shape[1] + gen_kwargs["min_length"]
    else:
        gen_kwargs_ = gen_kwargs

    if device is None:
        if model.config.is_encoder_decoder:
            # seq2seq LM
            device = model.encoder.first_device
        else:
            # causal LM
            device = model.transformer.first_device

    # generate
    gen_output = model.generate(
        inputs=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,  # TODO
        **gen_kwargs_,
    )

    # number of tokens generated
    seq_length = len(gen_output["scores"])

    # get only the generated text (excluding prompt)
    gen_tokens = gen_output["sequences"][:, -seq_length:]

    # to texts

    gen_texts = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in gen_tokens.tolist()
    ]

    return gen_texts
