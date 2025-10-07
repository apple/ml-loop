#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

import torch
from cut_cross_entropy import LinearCrossEntropyImpl, linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from torch import FloatTensor, Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from phi_agents.utils.cuda import cuda_arc_to_major, get_cuda_architecture
from phi_agents.utils.logger import get_phi_logger

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutputWithPast

logger = get_phi_logger()


@dataclass
class EfficientLMOutput(CausalLMOutputWithPast):  # type: ignore
    sampled_logprobs: torch.Tensor | None = None


def memory_efficient_forward(
    self: Any,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Any,
) -> EfficientLMOutput:
    # input tensor of shape [b, s]
    assert input_ids is not None

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    h: FloatTensor = cast(FloatTensor, outputs.last_hidden_state)
    assert isinstance(h, Tensor), f"{h=} is not a tensor"

    is_float16 = h.dtype == torch.bfloat16 or h.dtype == torch.float16

    temperature = kwargs.get("temperature", None)
    if temperature is not None and temperature != 1.0:
        h = h / temperature

    targets = input_ids.clone()
    if get_cuda_architecture() > cuda_arc_to_major["volta"] and is_float16:
        impl = LinearCrossEntropyImpl.CCE_EXACT
    else:
        impl = LinearCrossEntropyImpl.TORCH_COMPILE
        targets = targets.to(torch.int64)

    is_output = kwargs["is_output"]
    if is_output is not None:
        targets[~is_output] = IGNORE_INDEX

    # If you're using the debugger & the torch compile version and this fails to compile try
    # running the again but with the environment variable TORCHDYNAMO_DISABLE=1 set to disable
    # compile.
    lm_head: nn.Linear = self.lm_head
    weight = lm_head.weight
    output = linear_cross_entropy(
        e=h,
        c=weight,
        targets=targets,
        shift=True,
        reduction="none",
        impl=impl,
    )

    return EfficientLMOutput(
        loss=None,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=(h,),
        attentions=outputs.attentions,
        sampled_logprobs=output,
    )


def enable_memory_efficient_forward(llm: Any) -> None:  # `llm` is something like `Qwen2ForCausalLM`
    llm.forward = MethodType(memory_efficient_forward, llm)
