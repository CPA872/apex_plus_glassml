from functools import lru_cache
from typing import List, Tuple

import pandas as pd

from apex_plus.ir.tasks.attention import MHAHead
from apex_plus.ir.tasks.ffn import MLPFilter, GLUFilter, SwiGLUFilter
from apex_plus.utils.dtype import DTYPE, dtype_to_str

MAX_NUM_INPUT_TOKENS = 64 * 1024  # Max profiled n dimension for GEMM
MAX_PROFILED_SEQLEN = 16 * 1024   # Max profiled seq_len for attention

# Profile tables live at profile/comp/{gpu}/{op}.csv. Each file is one
# max-clock profile per GPU; frequency is recorded as a column but not used
# for lookup.
TMPL = "profile/comp/{gpu}/{op}.csv"


@lru_cache(maxsize=512)
def _load_table(op_kind: str, gpu: str) -> pd.DataFrame:
    """op_kind ∈ {'gemm', 'attn', 'bimha'}"""
    return pd.read_csv(TMPL.format(gpu=gpu, op=op_kind))


def _gemm_df(gpu: str) -> pd.DataFrame:
    return _load_table("gemm", gpu)


def _attn_df(gpu: str) -> pd.DataFrame:
    return _load_table("attn", gpu)


def _mla_df(gpu: str) -> pd.DataFrame:
    return _load_table("flash_mla", gpu)


def _interpolate(
    df: pd.DataFrame,
    col1: str,
    col1_val: int,
    col2: str,
    col2_val: int,
    target_col: str,
) -> float:
    small = df[(df[col1] <= col1_val) & (df[col2] <= col2_val)]
    large = df[(df[col1] >= col1_val) & (df[col2] >= col2_val)]
    if len(small) == 0 or len(large) == 0:
        if len(small) == 0 and len(large) != 0:
            return large.iloc[0][target_col]
        if len(large) == 0 and len(small) != 0:
            return small.iloc[-1][target_col]
        else:
            raise ValueError(
                "Cannot interpolate. "
                f"col1: {col1}, col1_val: {col1_val}, "
                f"col2: {col2}, col2_val: {col2_val}."
            )

    small = small.iloc[-1]
    large = large.iloc[0]
    if small[col1] == large[col1] and small[col2] == large[col2]:
        return small[target_col]
    elif small[col1] == large[col1]:
        r2 = (col2_val - small[col2]) / (large[col2] - small[col2])
        return small[target_col] * (1 - r2) + large[target_col] * r2
    elif small[col2] == large[col2]:
        r1 = (col1_val - small[col1]) / (large[col1] - small[col1])
        return small[target_col] * (1 - r1) + large[target_col] * r1
    else:
        r1 = (col1_val - small[col1]) / (large[col1] - small[col1])
        r2 = (col2_val - small[col2]) / (large[col2] - small[col2])
        r = (r1 * r2) ** 0.5
        return small[target_col] * (1 - r) + large[target_col] * r


@lru_cache(maxsize=512)
def _gemm_time(
    gpu: str,
    m: int,
    k: int,
    n: int,
    dtype: str,
) -> Tuple[float, float]:
    df = _gemm_df(gpu)
    # bf16/fp16 have identical Tensor-Core throughput on Hopper/Blackwell.
    # Fall back to the other if the requested dtype isn't profiled.
    candidates = [dtype]
    if dtype == "bfloat16" and not (df["dtype"] == "bfloat16").any():
        candidates.append("half")
    elif dtype == "half" and not (df["dtype"] == "half").any():
        candidates.append("bfloat16")
    for d in candidates:
        df_d = df[df["dtype"] == d]
        df_d = df_d[df_d["n"] == n]
        if not df_d.empty:
            exe_time = _interpolate(df_d, "m", m, "k", k, "time(us)")
            if "avg_energy(uJ)" in df_d.columns:
                exe_energy = _interpolate(df_d, "m", m, "k", k, "avg_energy(uJ)")
            else:
                exe_energy = 0.0
            return exe_time, exe_energy
    raise AssertionError(
        f"Cannot find gemm time for {gpu}, dtype in {candidates}, "
        f"m={m}, k={k}, n={n}"
    )


def round_to_power_of_2(n):
    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n
    # Find the closest power of 2 greater than or equal to n
    power_of_2_greater = 1
    while power_of_2_greater < n:
        power_of_2_greater <<= 1
    # Find the closest power of 2 less than n
    power_of_2_less = power_of_2_greater >> 1
    # Return the closest power of 2
    if (n - power_of_2_less) < (power_of_2_greater - n):
        return power_of_2_less
    else:
        return power_of_2_greater


def gemm_time(
    gpu: str,
    m: int,
    k: int,
    n: int,
    dtype: str,
) -> Tuple[float, float]:
    # Round up to the nearest multiple of 128.
    n = (n + 127) // 128 * 128 if n > 64 else 64

    if dtype == "float8":
        m = 16 if m < 16 else m
        k = 16 if k < 16 else k
        m = round_to_power_of_2(m)
        k = round_to_power_of_2(k)
        n = round_to_power_of_2(n)

    # Extrapolate linearly for n beyond profiled range.
    # For large n, GEMMs are compute-bound and scale linearly.
    if n > MAX_NUM_INPUT_TOKENS:
        scale = n / MAX_NUM_INPUT_TOKENS
        time, energy = _gemm_time(gpu, m, k, MAX_NUM_INPUT_TOKENS, dtype)
        return time * scale, energy * scale

    return _gemm_time(gpu, m, k, n, dtype)


@lru_cache(maxsize=512)
def attn_kernel_time(
    gpu: str,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: str,
    causal: bool = True,
) -> Tuple[float, float]:
    """Look up FlashAttention kernel time from profile/comp/{gpu}/attn.csv.

    Filters by (dtype, head_size, num_kv_heads, causal, seq_len), then
    interpolates on (batch_size, num_heads). For seq_len past the profiled
    max (16K), extrapolates quadratically since causal attention compute is
    O(B·H·L²·D).
    """
    df = _attn_df(gpu)
    df = df[df["dtype"] == dtype]
    df = df[df["head_size"] == head_size]
    df = df[df["num_heads_kv"] == num_kv_heads]
    df = df[df["causal"] == int(causal)]
    assert not df.empty, (
        f"Cannot find attn time for {gpu}, {dtype}, head_size={head_size}, "
        f"num_kv_heads={num_kv_heads}, causal={causal}"
    )

    # Round up to nearest multiple of 16 (matches profiler grid).
    seq_len = (seq_len + 15) // 16 * 16

    # Quadratic L extrapolation past max profiled seq_len.
    if seq_len > MAX_PROFILED_SEQLEN:
        scale = (seq_len / MAX_PROFILED_SEQLEN) ** 2
        df = df[df["seq_len"] == MAX_PROFILED_SEQLEN]
        t = _interpolate(df, "batch_size", batch_size, "num_heads", num_heads, "time(us)")
        e = _interpolate(df, "batch_size", batch_size, "num_heads", num_heads, "avg_energy(uJ)")
        return t * scale, e * scale

    df = df[df["seq_len"] == seq_len]
    assert not df.empty, (
        f"Cannot find attn time for {gpu}, seq_len={seq_len} after snap"
    )
    exe_time = _interpolate(
        df, "batch_size", batch_size, "num_heads", num_heads, "time(us)"
    )
    exe_energy = _interpolate(
        df, "batch_size", batch_size, "num_heads", num_heads, "avg_energy(uJ)"
    )
    return exe_time, exe_energy


@lru_cache(maxsize=512)
def bi_attn_kernel_time(
    gpu: str,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: str,
) -> Tuple[float, float]:
    """Bidirectional (encoder) attention — uses attn.csv with causal=0.

    Currently the FA profiler runs causal-only by default, so this will
    fail for any encoder query unless attn.csv is extended. Kept for API
    completeness; not exercised by the 5 supported decoder/MoE models.
    """
    return attn_kernel_time(
        gpu, head_size, num_heads, num_kv_heads,
        batch_size, seq_len, dtype, causal=False,
    )


@lru_cache(maxsize=512)
def mla_kernel_time(
    gpu: str,
    num_heads_q: int,
    batch_size: int,
    cache_seqlen: int,
    dtype: str,
) -> Tuple[float, float]:
    """FlashMLA decode-mode kernel time from profile/comp/{gpu}/flash_mla.csv.

    Filters by (dtype, num_heads_q), interpolates on (batch_size, cache_seqlen).
    Energy is not measured by the FlashMLA profile, so returns 0.0.
    """
    df = _mla_df(gpu)
    df = df[df["dtype"] == dtype]
    df = df[df["num_heads_q"] == num_heads_q]
    assert not df.empty, (
        f"Cannot find flash_mla time for {gpu}, {dtype}, "
        f"num_heads_q={num_heads_q}"
    )
    exe_time = _interpolate(
        df, "batch_size", batch_size,
        "cache_seqlen", cache_seqlen, "time(us)",
    )
    return exe_time, 0.0


def _mla_time(
    gpu: str,
    heads,  # List[MLAHead]
    dtype: DTYPE,
    input_lens: List[int],
    cached_lens: List[int],
    masked: bool,
) -> Tuple[float, float]:
    """Time one MLA layer: Q-down, Q-up, KV-down (with rope), attention,
    O-proj. Prefill attention uses flash_attn at D=qk_nope+qk_rope=192 with
    n_kv=1 as a proxy (FlashMLA kernel only ships decode mode publicly).
    Decode attention uses the FlashMLA lookup directly.
    """
    head = heads[0]
    n_q_per_dev = len(heads)
    hidden = head.hidden_size
    q_lora = head.q_lora_rank
    kv_lora = head.kv_lora_rank
    qk_nope = head.qk_nope_head_dim
    qk_rope = head.qk_rope_head_dim
    v_dim = head.v_head_dim
    d_k = qk_nope + qk_rope          # = 192 for DSv3 / Kimi
    d_kv = kv_lora + qk_rope         # = 576 for DSv3 / Kimi (KV-down output dim)

    dtype_str = dtype_to_str(dtype)
    atten_dtype = "half" if dtype == DTYPE.FLOAT8 else dtype_str
    num_total_input_tokens = sum(input_lens)
    total_time = 0.0
    total_energy = 0.0

    # 1. Q-down: hidden -> q_lora_rank
    t, e = gemm_time(
        gpu=gpu, m=q_lora, k=hidden, n=num_total_input_tokens, dtype=dtype_str,
    )
    total_time += t; total_energy += e

    # 2. Q-up: q_lora_rank -> n_q_per_dev * (qk_nope + qk_rope)
    t, e = gemm_time(
        gpu=gpu, m=n_q_per_dev * d_k, k=q_lora,
        n=num_total_input_tokens, dtype=dtype_str,
    )
    total_time += t; total_energy += e

    # 3. KV-down + rope: hidden -> kv_lora_rank + qk_rope_head_dim
    t, e = gemm_time(
        gpu=gpu, m=d_kv, k=hidden, n=num_total_input_tokens, dtype=dtype_str,
    )
    total_time += t; total_energy += e

    # 4. Prompt attention via flash_attn(D=192, n_kv=1) proxy.
    # Note: this overstates by ~25% because MLA's V dim is 128, not 192;
    # the V-AV portion of attention compute is proportionally smaller.
    prompt_indices = [i for i in range(len(input_lens)) if cached_lens[i] == 0]
    if prompt_indices:
        prompt_b = len(prompt_indices)
        prompt_l = sum(input_lens[i] for i in prompt_indices) / prompt_b
        att_t, att_e = attn_kernel_time(
            gpu=gpu, head_size=d_k, num_heads=n_q_per_dev,
            num_kv_heads=1, batch_size=prompt_b,
            seq_len=prompt_l, dtype=atten_dtype, causal=masked,
        )
        if dtype == DTYPE.FLOAT8:
            att_t /= 1.5
        total_time += att_t; total_energy += att_e

    # 5. Decode attention via FlashMLA lookup.
    decode_indices = [i for i in range(len(input_lens)) if cached_lens[i] > 0]
    if decode_indices:
        decode_b = len(decode_indices)
        avg_cache_l = sum(cached_lens[i] for i in decode_indices) / decode_b
        att_t, att_e = mla_kernel_time(
            gpu=gpu, num_heads_q=n_q_per_dev,
            batch_size=decode_b, cache_seqlen=int(avg_cache_l),
            dtype=atten_dtype,
        )
        total_time += att_t; total_energy += att_e

    # 6. O-proj: n_q_per_dev * v_dim -> hidden
    t, e = gemm_time(
        gpu=gpu, m=hidden, k=n_q_per_dev * v_dim,
        n=num_total_input_tokens, dtype=dtype_str,
    )
    total_time += t; total_energy += e

    return total_time, total_energy


def attn_time(
    gpu: str,
    heads,  # List[MHAHead] | List[BiMHAHead] | List[MQAHead] | List[MLAHead]
    dtype: DTYPE,
    input_lens: List[int],
    cached_lens: List[int],
    masked: bool,  # True for MHA/MQA/MLA, False for BiMHA
) -> Tuple[float, float]:
    if not heads:
        return 0.0, 0.0
    if not input_lens:
        return 0.0, 0.0

    # MLA has a different weight decomposition (LoRA + latent KV); route to
    # its specialized timing function.
    if heads[0].__class__.__name__ == "MLAHead":
        return _mla_time(gpu, heads, dtype, input_lens, cached_lens, masked)

    dtype_str = dtype_to_str(dtype)
    # FA profile is bf16/half only. Synthetic FP8 = half / 1.5.
    atten_dtype = "half" if dtype == DTYPE.FLOAT8 else dtype_to_str(dtype)
    num_heads = len(heads)
    head_size = heads[0].head_size
    hidden_size = heads[0].hidden_size
    # For GQA/MQA, K and V have fewer heads than Q. Fused QKV linear has
    # output dim = (n_q + 2*n_kv) * head_size. Falls back to n_kv = n_q
    # for plain MHA where kv_head_id is absent.
    if hasattr(heads[0], "kv_head_id"):
        num_kv_heads = len(set(t.kv_head_id for t in heads))
    else:
        num_kv_heads = num_heads
    qkv_out_dim = (num_heads + 2 * num_kv_heads) * head_size

    num_total_input_tokens = sum(input_lens)
    total_time = 0.0
    total_energy = 0.0
    # 1. QKV Linear
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        m=qkv_out_dim,
        k=hidden_size,
        n=num_total_input_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy

    # 2. Prompt attention
    prompt_indices = [i for i in range(len(input_lens)) if cached_lens[i] == 0]
    if prompt_indices:
        prompt_batch_size = len(prompt_indices)
        prompt_seq_len = sum(input_lens[i] for i in prompt_indices) / prompt_batch_size

        attention_time, attention_energy = attn_kernel_time(
            gpu=gpu,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            batch_size=prompt_batch_size,
            seq_len=prompt_seq_len,
            dtype=atten_dtype,
            causal=masked,
        )
        # Synthetic FP8 = half / 1.5
        attention_time = (
            (attention_time / 1.5) if dtype == DTYPE.FLOAT8 else attention_time
        )
        total_time += attention_time
        total_energy += attention_energy

    # 3. Cached / decode attention
    decoding_indices = [i for i in range(len(input_lens)) if cached_lens[i] > 0]
    if decoding_indices:
        decoding_batch_size = len(decoding_indices)
        decoding_seq_len = 1
        exe_time, exe_energy = attn_kernel_time(
            gpu=gpu,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            batch_size=decoding_batch_size,
            seq_len=decoding_seq_len,
            dtype=atten_dtype,
            causal=masked,
        )
        total_time += exe_time
        total_energy += exe_energy

    # 4. Output Linear
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        m=hidden_size,
        k=num_heads * head_size,
        n=num_total_input_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy


def mlp_time(
    gpu: str,
    filters: List[MLPFilter],
    dtype: DTYPE,
    num_tokens: int,
) -> Tuple[float, float]:
    if not filters:
        return 0.0, 0.0

    num_filters = len(filters)
    hidden_size = filters[0].hidden_size
    dtype_str = dtype_to_str(dtype)

    # 1. MLP 0
    total_time, total_energy = gemm_time(
        gpu=gpu, m=num_filters, k=hidden_size, n=num_tokens, dtype=dtype_str,
    )
    # 2. MLP 1
    exe_time, exe_energy = gemm_time(
        gpu=gpu, m=hidden_size, k=num_filters, n=num_tokens, dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy


def glu_time(
    gpu: str,
    filters: List[GLUFilter],
    dtype: DTYPE,
    num_tokens: int,
) -> Tuple[float, float]:
    if not filters:
        return 0.0, 0.0

    num_filters = len(filters)
    hidden_size = filters[0].hidden_size
    dtype_str = dtype_to_str(dtype)

    # 1. MLP 0 (gate + up fused via 2x output)
    total_time, total_energy = gemm_time(
        gpu=gpu, m=num_filters * 2, k=hidden_size, n=num_tokens, dtype=dtype_str,
    )
    # 2. MLP 1 (down)
    exe_time, exe_energy = gemm_time(
        gpu=gpu, m=hidden_size, k=num_filters, n=num_tokens, dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy


def swiglu_time(
    gpu: str,
    filters,
    dtype: DTYPE,
    num_tokens: int,
) -> Tuple[float, float]:
    if not filters:
        return 0.0, 0.0

    num_filters = len(filters)
    hidden_size = filters[0].hidden_size
    dtype_str = dtype_to_str(dtype)

    # 1. gate
    total_time, total_energy = gemm_time(
        gpu=gpu, m=num_filters, k=hidden_size, n=num_tokens, dtype=dtype_str,
    )
    # 2. up
    exe_time, exe_energy = gemm_time(
        gpu=gpu, m=hidden_size, k=num_filters, n=num_tokens, dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    # 3. down
    exe_time, exe_energy = gemm_time(
        gpu=gpu, m=num_filters, k=hidden_size, n=num_tokens, dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy
