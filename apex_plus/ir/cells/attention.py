from typing import List, Union

from apex_plus.ir.cell import Cell
from apex_plus.ir.tasks.attention import MHAHead, MQAHead, BiMHAHead, MLAHead
from apex_plus.ir.tasks.ffn import MLPFilter


class MHA(Cell):  # Masked, unidirectional MHA used in decoders

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.heads = [
            MHAHead(i, self.head_size, self.hidden_size) for i in range(self.num_heads)
        ]

    def get_tasks(self) -> List[MHAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MHA):
            return False
        return (
            self.num_heads == other.num_heads and self.hidden_size == other.hidden_size
        )


class BiMHA(Cell):  # Unmasked, bidirectional MHA used in encoders

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.heads = [
            BiMHAHead(i, self.head_size, self.hidden_size)
            for i in range(self.num_heads)
        ]

    def get_tasks(self) -> List[BiMHAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, BiMHAHead):
            return False
        return (
            self.num_heads == other.num_heads and self.hidden_size == other.hidden_size
        )


class ParallelMHAMLP(Cell):

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.heads = [
            MHAHead(i, self.head_size, self.hidden_size) for i in range(self.num_heads)
        ]
        self.filters = [
            MLPFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.hidden_size)
        ]

    def get_tasks(self) -> List[Union[MHAHead, MLPFilter]]:
        return self.heads + self.filters

    def get_num_task_types(self) -> int:
        return 2

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, ParallelMHAMLP):
            return False
        return (
            self.num_heads == other.num_heads
            and self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )


class MQA(Cell):

    def __init__(
        self,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size = hidden_size

        assert self.num_query_heads % self.num_kv_heads == 0
        self.num_query_per_kv = self.num_query_heads // self.num_kv_heads
        self.heads = [
            MQAHead(
                i,
                i // self.num_query_per_kv,
                self.head_size,
                self.hidden_size,
            )
            for i in range(self.num_query_heads)
        ]

    def get_tasks(self) -> List[MQAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MQA):
            return False
        return (
            self.num_query_heads == other.num_query_heads
            and self.num_kv_heads == other.num_kv_heads
            and self.head_size == other.head_size
            and self.hidden_size == other.hidden_size
        )


class MLA(Cell):
    """DeepSeek-V3 / Kimi-K2 Multi-head Latent Attention.

    Each query head is one MLAHead task. Q is decomposed via q_lora_rank;
    KV is a single shared latent of dim kv_lora_rank with an extra
    qk_rope_head_dim for the rotary part. K/V up-projections are absorbed
    into the per-head Q-up and into the O-proj input respectively.
    """

    def __init__(
        self,
        num_query_heads: int,
        hidden_size: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ) -> None:
        self.num_query_heads = num_query_heads
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # Effective Q/K head dim (= qk_nope + qk_rope).
        self.head_size = qk_nope_head_dim + qk_rope_head_dim

        self.heads = [
            MLAHead(
                head_id=i,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
            )
            for i in range(num_query_heads)
        ]

    def get_tasks(self) -> List[MLAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        return (
            isinstance(other, MLA)
            and self.num_query_heads == other.num_query_heads
            and self.hidden_size == other.hidden_size
            and self.q_lora_rank == other.q_lora_rank
            and self.kv_lora_rank == other.kv_lora_rank
            and self.qk_nope_head_dim == other.qk_nope_head_dim
            and self.qk_rope_head_dim == other.qk_rope_head_dim
            and self.v_head_dim == other.v_head_dim
        )


class ParallelMQAMLP(Cell):

    def __init__(
        self,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        assert self.num_query_heads % self.num_kv_heads == 0
        self.num_query_per_kv = self.num_query_heads // self.num_kv_heads
        self.heads = [
            MQAHead(
                i,
                i // self.num_query_per_kv,
                self.head_size,
                self.hidden_size,
            )
            for i in range(self.num_query_heads)
        ]
        self.filters = [
            MLPFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.hidden_size)
        ]

    def get_tasks(self) -> List[Union[MQAHead, MLPFilter]]:
        return self.heads + self.filters

    def get_num_task_types(self) -> int:
        return 2

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, ParallelMQAMLP):
            return False
        return (
            self.num_query_heads == other.num_query_heads
            and self.num_kv_heads == other.num_kv_heads
            and self.head_size == other.head_size
            and self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )
