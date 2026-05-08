from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel
from apex_plus.ir.cells.attention import MQA
from apex_plus.ir.cells.ffn import SwiGLU
from apex_plus.ir.block import Block


class Qwen3(ApexModel):
    """Dense Qwen3 (e.g. Qwen3-32B). Same attention as Qwen3-MoE
    (GQA + decoupled `head_dim`) but FFN is a single SwiGLU rather
    than a top-K MoE.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    @classmethod
    def from_hf(cls, config) -> "Qwen3":
        head_size = getattr(config, "head_dim", None)
        if head_size is None:
            head_size = config.hidden_size // config.num_attention_heads
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_size=head_size,
        )

    def to_ir(self) -> Transformer:
        mqa = MQA(
            self.num_heads, self.num_kv_heads, self.head_size, self.hidden_size
        )
        swiglu = SwiGLU(self.hidden_size, self.intermediate_size)
        decoder_block = Block(cells=[mqa, swiglu])
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
            f"Qwen3(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_size={self.head_size})"
        )
