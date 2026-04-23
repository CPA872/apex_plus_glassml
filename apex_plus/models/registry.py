from typing import Optional

from transformers import AutoConfig

from apex_plus.ir.transformer import Transformer
from apex_plus.models.bloom import Bloom
from apex_plus.models.gpt_bigcode import GPTBigCode
from apex_plus.models.gpt_neox import GPTNeoX
from apex_plus.models.gpt2 import GPT2
from apex_plus.models.gptj import GPTJ
from apex_plus.models.llama import LLaMA
from apex_plus.models.moe import OPTMoE
from apex_plus.models.opt import OPT
from apex_plus.models.whisper import Whisper
from apex_plus.models.CLIP_vision import CLIPVision
from apex_plus.models.mistral import Mistral
from apex_plus.models.mixtral import Mixtral
from apex_plus.models.t5 import T5
from apex_plus.models.llama3 import LLaMA3

_MODEL_REGISTRY = {
    "BloomForCausalLM": Bloom,
    "GPT2LMHeadModel": GPT2,
    "GPTBigCodeForCausalLM": GPTBigCode,
    "GPTJForCausalLM": GPTJ,
    "GPTNeoXForCausalLM": GPTNeoX,
    "LlamaForCausalLM": LLaMA,
    "OPTForCausalLM": OPT,
    "WhisperForConditionalGeneration": Whisper,
    "CLIPModel": CLIPVision,
    "MistralForCausalLM": Mistral,
    "MixtralForCausalLM": Mixtral,
    "T5ForConditionalGeneration": T5,
    "Llama3ForCausalLM": LLaMA3,
}


def get_model_registry():
    return _MODEL_REGISTRY


def get_model_ir(
    model_name: str,
    num_experts: Optional[int],
    topk: int,
    capacity_factor: float,
) -> Transformer:
    config = AutoConfig.from_pretrained(model_name)
    if len(config.architectures) > 1:
        raise ValueError("Only single architecture models are supported")

    arch = config.architectures[0]
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {arch} not supported")

    # Native MoE architectures that read expert config from HF config.json
    _NATIVE_MOE_ARCHS = {"MixtralForCausalLM"}
    # Dense architectures that support conversion to MoE via extra args
    _MOE_CONVERTIBLE_ARCHS = {"OPTForCausalLM": OPTMoE}

    if arch in _NATIVE_MOE_ARCHS:
        model = _MODEL_REGISTRY[arch].from_hf(
            config,
            num_experts=getattr(config, "num_local_experts", 8),
            topk=getattr(config, "num_experts_per_tok", 2),
            capacity_factor=capacity_factor,
        )
    elif num_experts is not None and num_experts > 1:
        # Dense model converted to MoE.
        if arch not in _MOE_CONVERTIBLE_ARCHS:
            raise ValueError(
                f"Architecture {arch} does not support dense-to-MoE conversion. "
                f"Supported: {list(_MOE_CONVERTIBLE_ARCHS.keys())}"
            )
        model = _MOE_CONVERTIBLE_ARCHS[arch].from_hf(
            config, num_experts=num_experts, topk=topk, capacity_factor=capacity_factor
        )
    else:
        # Non-MoE models.
        model = _MODEL_REGISTRY[arch].from_hf(config)

    return model.to_ir(), model
