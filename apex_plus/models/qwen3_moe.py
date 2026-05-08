from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel
from apex_plus.ir.cells.attention import MQA
from apex_plus.ir.cells.ffn import SwiMoE
from apex_plus.ir.block import Block


class Qwen3Moe(ApexModel):

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        num_experts: int,
        topk: int,
        capacity_factor: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_experts = num_experts
        self.topk = topk
        self.capacity_factor = capacity_factor

        if num_experts <= topk:
            raise ValueError(
                f"num_experts {num_experts} must be larger than topk {topk}."
            )

    @classmethod
    def from_hf(
        cls,
        config,
        num_experts: int,
        topk: int,
        capacity_factor: float,
    ) -> "Qwen3Moe":
        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if mlp_only_layers:
            raise NotImplementedError(
                "Qwen3 MoE configs with non-empty `mlp_only_layers` (mixed "
                "dense + MoE layer stacks) are not yet modeled. Got: "
                f"{mlp_only_layers}"
            )
        sparse_step = getattr(config, "decoder_sparse_step", 1)
        if sparse_step != 1:
            raise NotImplementedError(
                f"decoder_sparse_step={sparse_step}: only every-layer MoE "
                "(decoder_sparse_step=1) is currently modeled."
            )

        head_size = getattr(config, "head_dim", None)
        if head_size is None:
            head_size = config.hidden_size // config.num_attention_heads
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_size=head_size,
            num_experts=num_experts,
            topk=topk,
            capacity_factor=capacity_factor,
        )

    def to_ir(self) -> Transformer:
        mqa = MQA(
            self.num_heads, self.num_kv_heads, self.head_size, self.hidden_size
        )
        swimoe = SwiMoE(
            self.num_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            self.topk,
            self.capacity_factor,
        )
        decoder_block = Block(cells=[mqa, swimoe])
        return Transformer.from_blocks(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_blocks=0,
            encoder_block=None,
            num_decoder_blocks=self.num_layers,
            decoder_block=decoder_block,
        )

    def __repr__(self) -> str:
        return (
            f"Qwen3Moe(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"moe_intermediate_size={self.moe_intermediate_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_size={self.head_size}, "
            f"num_experts={self.num_experts}, "
            f"topk={self.topk}, "
            f"capacity_factor={self.capacity_factor})"
        )
