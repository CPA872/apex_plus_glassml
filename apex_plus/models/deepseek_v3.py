from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel
from apex_plus.ir.cells.attention import MLA
from apex_plus.ir.cells.ffn import SwiMoE
from apex_plus.ir.block import Block


class DeepseekV3(ApexModel):
    """DeepSeek-V3 / Kimi-K2 model class. Both share the same DSv3
    architecture (MLA attention + MoE FFN with SwiGLU experts).

    Notes / approximations:
    - DSv3 has `first_k_dense_replace=3` (first 3 layers are dense). We
      treat all layers as MoE — small approximation for a 61-layer model.
    - DSv3 has `n_shared_experts=1` (one shared expert always activated).
      Not modeled separately — folded into the routed expert count.
    - Multi-Token Prediction (MTP) layer is not modeled.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        num_q_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        num_experts: int,
        topk: int,
        moe_intermediate_size: int,
        capacity_factor: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_experts = num_experts
        self.topk = topk
        self.moe_intermediate_size = moe_intermediate_size
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
    ) -> "DeepseekV3":
        # DSv3 stores routed-expert count under n_routed_experts; the registry
        # caller passes whatever it found. Prefer config field if present.
        ne = getattr(config, "n_routed_experts", num_experts)
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            num_experts=ne,
            topk=topk,
            moe_intermediate_size=config.moe_intermediate_size,
            capacity_factor=capacity_factor,
        )

    def to_ir(self) -> Transformer:
        mla = MLA(
            num_query_heads=self.num_q_heads,
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )
        moe = SwiMoE(
            self.num_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            self.topk,
            self.capacity_factor,
        )
        decoder_block = Block(cells=[mla, moe])
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
            f"DeepseekV3(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"num_q_heads={self.num_q_heads}, "
            f"q_lora_rank={self.q_lora_rank}, "
            f"kv_lora_rank={self.kv_lora_rank}, "
            f"qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, "
            f"v_head_dim={self.v_head_dim}, "
            f"num_experts={self.num_experts}, "
            f"topk={self.topk}, "
            f"moe_intermediate_size={self.moe_intermediate_size}, "
            f"capacity_factor={self.capacity_factor})"
        )
