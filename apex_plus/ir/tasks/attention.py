from typing import List

from apex_plus.ir.task import Task
from apex_plus.utils.dtype import DTYPE


class MHAHead(Task):

    def __init__(
        self,
        head_id: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.head_id = head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    def __repr__(self) -> str:
        return f"MHAHead(head_id={self.head_id})"

    @staticmethod
    def get_param_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        head = tasks[0]
        cnt = 4 * head.hidden_size * head.head_size
        return len(tasks) * cnt * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        head = tasks[0]
        num_heads = len(tasks)
        return 2 * num_heads * head.head_size * dtype.size

    @classmethod
    def is_attn(cls) -> bool:
        return True


class BiMHAHead(Task):

    def __init__(
        self,
        head_id: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.head_id = head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    def __repr__(self) -> str:
        return f"MHAHead(head_id={self.head_id})"

    @staticmethod
    def get_param_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        head = tasks[0]
        cnt = 4 * head.hidden_size * head.head_size
        return len(tasks) * cnt * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        # Assume no kv cache for bidirectional MHA as it's not used in decoders
        return 0

    @classmethod
    def is_attn(cls) -> bool:
        return True


class MQAHead(Task):

    def __init__(
        self,
        query_head_id: int,
        kv_head_id: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.query_head_id = query_head_id
        self.kv_head_id = kv_head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    def __repr__(self) -> str:
        return (
            f"MQAHead(query_head_id={self.query_head_id}, "
            f"kv_head_id={self.kv_head_id})"
        )

    @staticmethod
    def get_param_size(tasks: List["MQAHead"], dtype: DTYPE) -> int:
        num_query_heads = len(tasks)
        # Different query heads might share the same KV heads.
        num_kv_heads = len(set(task.kv_head_id for task in tasks))
        head_size = tasks[0].head_size
        hidden_size = tasks[0].hidden_size

        q = num_query_heads * head_size * hidden_size
        k = num_kv_heads * head_size * hidden_size
        v = num_kv_heads * head_size * hidden_size
        o = num_query_heads * head_size * hidden_size
        cnt = q + k + v + o
        return cnt * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MQAHead"], dtype: DTYPE) -> int:
        num_kv_heads = len(set(task.kv_head_id for task in tasks))
        head_size = tasks[0].head_size
        return 2 * num_kv_heads * head_size * dtype.size

    @classmethod
    def is_attn(cls) -> bool:
        return True


class MLAHead(Task):
    """One query head of DeepSeek-V3 / Kimi-K2 Multi-head Latent Attention.

    MLA decomposes Q via a low-rank LoRA (q_lora_rank) and KV via a single
    shared latent (kv_lora_rank), with the per-head Q/K dim split into
    no-position-encoding (qk_nope) and rotary (qk_rope) parts. KV cache
    stores only the compressed latent, hence H_KV=1 from the simulator's
    perspective.

    `head_size` is set to qk_nope + qk_rope (=192 for both DSv3 and Kimi)
    so that downstream code expecting a head dim sees the right value.
    """

    def __init__(
        self,
        head_id: int,
        hidden_size: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ) -> None:
        self.head_id = head_id
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # Compatibility: downstream (simulator, profile lookups) keys on
        # `head_size` for attention. For MLA the effective Q/K dim is
        # qk_nope + qk_rope.
        self.head_size = qk_nope_head_dim + qk_rope_head_dim

    def __repr__(self) -> str:
        return f"MLAHead(head_id={self.head_id})"

    @staticmethod
    def get_param_size(tasks: List["MLAHead"], dtype: DTYPE) -> int:
        # Replicated weights (Q-down, KV-down) are counted once per device
        # since they're not split by query head. Per-head weights are Q-up,
        # V-up (as part of the absorbed KV chain), and the O-proj input slice.
        n = len(tasks)
        head = tasks[0]
        d_k = head.qk_nope_head_dim + head.qk_rope_head_dim
        d_kv = head.kv_lora_rank + head.qk_rope_head_dim

        q_down = head.hidden_size * head.q_lora_rank
        kv_down = head.hidden_size * d_kv
        q_up_per_head = head.q_lora_rank * d_k
        v_up_per_head = head.kv_lora_rank * head.v_head_dim
        o_per_head = head.v_head_dim * head.hidden_size
        per_head = q_up_per_head + v_up_per_head + o_per_head

        return (q_down + kv_down + n * per_head) * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MLAHead"], dtype: DTYPE) -> int:
        # MLA caches only the compressed latent (kv_lora_rank + qk_rope),
        # NOT per-head — that's the whole point of the architecture.
        head = tasks[0]
        return (head.kv_lora_rank + head.qk_rope_head_dim) * dtype.size

    @classmethod
    def is_attn(cls) -> bool:
        return True
