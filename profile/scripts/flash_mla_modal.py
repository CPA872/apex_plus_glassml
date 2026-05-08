"""Modal launcher for FlashMLA decode profiling on H100 / B200.

Uses the vllm/vllm-openai image which ships flash_mla pre-built — no CUDA
build needed. Sweeps the absorbed-form MLA decode kernel across:

    H_Q ∈ {2, 4, 8, 16, 32, 64, 128}  — covers DeepSeek-V3 (n_q=128)
                                         and Kimi-K2 (n_q=64) under
                                         TP ∈ {1, 2, 4, 8, 16, 32, 64}
    B   ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}
    cache_seqlen ∈ {512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K}

Fixed (from DeepSeek-V3 / Kimi-K2 MLA architecture):
    D_K = 576  = kv_lora_rank (512) + qk_rope_head_dim (64)
    D_V = 512  = kv_lora_rank (512)
    H_KV = 1                              (single latent KV head)
    S_Q  = 1                              (decode mode)
    block_size = 64                       (paged KV)
    dtype = bfloat16

Output: flash_mla_shards/{gpu}_flash_mla.csv with schema:
    gpu, dtype, num_heads_q, batch_size, cache_seqlen, d_k, d_v, h_kv,
    s_q, block_size, time(us), bandwidth(GB/s)

Run:
    cd apex_plus/profile/scripts
    modal run flash_mla_modal.py --gpu "H100-SXM-80GB,B200-SXM-192GB"

Cost: ~$3-7 per GPU run, ~10-20 min wall.

NOTE: This profiles DECODE mode only (S_Q=1) — that's what the FlashMLA
kernel publicly supports today. For training-side prefill timing, fall back
to flash_attn(D=192) as a proxy with ~25% overstatement (V projection diff).
"""
import io
import os
import subprocess
import time
from typing import List, Tuple

import modal

app = modal.App("flash-mla-profile")

# vllm/vllm-openai bundles flash_mla via vllm.third_party.flashmla. No CUDA
# build needed on our side. The vllm image's bundled python is at
# /usr/bin/python3 (Modal needs add_python for harness Python).
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    .entrypoint([])
)

VLLM_PYTHON = "/usr/bin/python3"

# Sweep config — keep here as constants so the probe script (run in vllm's
# python, not ours) can read them via env vars.
H_Q_LIST = [2, 4, 8, 16, 32, 64, 128]
B_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CACHE_SEQLEN_LIST = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


# Probe runs *inside* the vllm container under VLLM_PYTHON. It reads the
# sweep spec from env, sweeps, and writes CSV bytes to stdout (we capture).
PROBE = r"""
import csv
import io
import os
import sys
import time

import torch
from vllm.third_party.flashmla.flash_mla_interface import (
    get_mla_metadata, flash_mla_with_kvcache,
)

# MLA absorbed-form constants for DeepSeek-V3 / Kimi-K2.
D_K = 576    # kv_lora_rank (512) + qk_rope_head_dim (64)
D_V = 512    # kv_lora_rank (512)
H_KV = 1
S_Q  = 1
BLOCK_SIZE = 64
DTYPE = torch.bfloat16

GPU_LABEL = os.environ["GPU_LABEL"]
H_Q_LIST = [int(x) for x in os.environ["H_Q_LIST"].split(",")]
B_LIST = [int(x) for x in os.environ["B_LIST"].split(",")]
L_LIST = [int(x) for x in os.environ["L_LIST"].split(",")]

NUM_WARMUP = 5
TARGET_WINDOW_S = 0.2  # adaptive iter loop target

def time_one(b, cache_seqlen, h_q):
    num_pages_per_req = (cache_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_pages = b * num_pages_per_req

    q = torch.randn(b, S_Q, h_q, D_K, device="cuda", dtype=DTYPE)
    kvcache = torch.randn(
        total_pages, BLOCK_SIZE, H_KV, D_K, device="cuda", dtype=DTYPE,
    )
    block_table = torch.arange(
        total_pages, device="cuda", dtype=torch.int32,
    ).reshape(b, num_pages_per_req)
    cache_seqlens = torch.full(
        (b,), cache_seqlen, device="cuda", dtype=torch.int32,
    )

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, S_Q * h_q // H_KV, H_KV,
    )

    def call():
        return flash_mla_with_kvcache(
            q, kvcache, block_table, cache_seqlens, D_V,
            tile_scheduler_metadata, num_splits, causal=True,
        )

    # warmups
    for _ in range(NUM_WARMUP):
        call()
    torch.cuda.synchronize()

    # adaptive iter count
    t0 = time.time()
    call()
    torch.cuda.synchronize()
    single_s = max(time.time() - t0, 1e-6)
    iters = max(5, min(1024, int(TARGET_WINDOW_S / single_s)))

    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    toc = time.time()
    time_us = (toc - tic) / iters * 1e6

    # Bandwidth: bytes read = b * cache_seqlen * H_KV * D_K * dtype_size
    bytes_read = b * cache_seqlen * H_KV * D_K * 2  # bf16
    bandwidth_gbs = (bytes_read / (time_us * 1e-6)) / 1e9 if time_us > 0 else 0.0

    del q, kvcache, block_table, cache_seqlens
    torch.cuda.empty_cache()
    return time_us, bandwidth_gbs


buf = io.StringIO()
w = csv.writer(buf)
w.writerow([
    "gpu", "dtype", "num_heads_q", "batch_size", "cache_seqlen",
    "d_k", "d_v", "h_kv", "s_q", "block_size",
    "time(us)", "bandwidth(GB/s)",
])

total = len(H_Q_LIST) * len(B_LIST) * len(L_LIST)
done = 0
for h_q in H_Q_LIST:
    for b in B_LIST:
        for cache_seqlen in L_LIST:
            done += 1
            try:
                t_us, bw = time_one(b, cache_seqlen, h_q)
            except Exception as e:
                print(
                    f"skip h_q={h_q} b={b} L={cache_seqlen}: {e}",
                    file=sys.stderr,
                )
                continue
            w.writerow([
                GPU_LABEL, "bfloat16", h_q, b, cache_seqlen,
                D_K, D_V, H_KV, S_Q, BLOCK_SIZE,
                int(t_us), f"{bw:.2f}",
            ])
            print(
                f"[{done}/{total}] h_q={h_q:>3} b={b:>3} L={cache_seqlen:>6}  "
                f"time={t_us:>9.2f}us  bw={bw:>7.1f} GB/s",
                file=sys.stderr,
            )

# Print CSV bytes to stdout (Modal captures it)
print(buf.getvalue(), end="")
"""


def _run_profile(gpu_label: str) -> bytes:
    env = {
        **os.environ,
        "GPU_LABEL": gpu_label,
        "H_Q_LIST": ",".join(str(x) for x in H_Q_LIST),
        "B_LIST": ",".join(str(x) for x in B_LIST),
        "L_LIST": ",".join(str(x) for x in CACHE_SEQLEN_LIST),
    }
    r = subprocess.run(
        [VLLM_PYTHON, "-c", PROBE],
        capture_output=True, text=True, env=env,
    )
    if r.stderr:
        print(r.stderr)
    if r.returncode != 0:
        r.check_returncode()
    return r.stdout.encode()


@app.function(image=image, gpu="H100", timeout=2 * 3600)
def profile_h100() -> bytes:
    return _run_profile("H100-SXM-80GB")


@app.function(image=image, gpu="B200", timeout=2 * 3600)
def profile_b200() -> bytes:
    return _run_profile("B200-SXM-192GB")


@app.local_entrypoint()
def main(
    gpu: str = "H100-SXM-80GB,B200-SXM-192GB",
    out_dir: str = "flash_mla_shards",
):
    """Run profile per GPU type, write one CSV per GPU."""
    import pathlib

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gpu_list = [g.strip() for g in gpu.split(",") if g.strip()]
    handles = []
    for g in gpu_list:
        if g == "H100-SXM-80GB":
            h = profile_h100.spawn()
        elif g == "B200-SXM-192GB":
            h = profile_b200.spawn()
        else:
            raise ValueError(f"Unsupported gpu '{g}'")
        handles.append((g, h))

    for g, h in handles:
        csv_bytes = h.get()
        path = out / f"{g}_flash_mla.csv"
        path.write_bytes(csv_bytes)
        print(f"wrote {path} ({len(csv_bytes)} bytes)")
