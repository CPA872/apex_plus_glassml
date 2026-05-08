"""Modal launcher for flash-attn profiling — subsampled sweep targeted at the
five models in apex_plus' SHORTCUT (Qwen3-30B-A3B, Qwen3-32B, Qwen3-235B,
DeepSeek-V3, Kimi-K2 — though MLA models will be served by flash_mla_modal).

One Modal container per (gpu, dtype, variant) — single GPU each, no Ray.
Wall-clock per run ~4-6 min on H100/B200; cost ~$0.30 each at typical Modal
rates. Output: <shards_dir>/<gpu>_<dtype>_<variant>.csv with the same schema
as profile/scripts/flash_attn.py.

Usage:
    cd apex_plus/profile/scripts
    modal run flash_attn_modal.py \\
        --gpu "H100-SXM-80GB,B200-SXM-192GB" \\
        --dtype bfloat16 --variant gqa
"""
import io
import os
import threading
import time
from typing import List, Tuple

import modal

from flash_attn_image import flash_attn_image

app = modal.App("flash-attn-profile")

# Ship flash_attn_image.py into the container so the import resolves remotely.
image = flash_attn_image.add_local_python_source("flash_attn_image")

NUM_WARMUP = 5
NUM_ITER = 100        # cap on iters per shape
MIN_ITER = 5          # floor when shapes take seconds each
TARGET_WINDOW_S = 0.2 # aim for ~200ms total measurement window

# Subsampled sweep axes — covers per-device shapes for the five Qwen3 / dense
# Qwen3 models under TP ∈ {1, 2, 4, 8, 16, 32, 64} and typical micro-batches.
# Edit these lists if your simulation needs different shapes.

# num_heads_q on one device (after TP sharding).
H = [1, 2, 4, 8, 16, 32, 64, 128]
# head_size — Qwen3 family uses 128; DSv3/Kimi MLA prefill proxy uses 192.
D = [128, 192]
# batch_size — typical micro-batch * DP / attn_DP.
B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# seq_len — multiples of 16, log-ish spacing dense in the 2K-16K range
# where most training queries land. Use quadratic L-extrapolation past 16K.
L = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384]

# B * L <= MAX_NUM_TOKENS keeps activations bounded and matches the original
# flash_attn.py constraint.
MAX_NUM_TOKENS = 16 * 4096

# GQA num_q / num_kv ratios. Includes high ratios for MLA prefill proxy
# where n_kv=1 (DSv3 H_Q=128 → ratio=128, Kimi H_Q=64 → ratio=64).
H_KV_RATIOS = [2, 4, 8, 16, 32, 64, 128]
# SWA window sizes — kept for completeness but no current model uses SWA.
WINDOWS = [128, 512, 2048, 4096]


@app.function(image=image, gpu="H100", timeout=2 * 3600)
def profile_h100(dtype: str, variant: str, causal: bool = True) -> bytes:
    return _profile_attn("H100-SXM-80GB", dtype, variant, causal)


@app.function(image=image, gpu="B200", timeout=2 * 3600)
def profile_b200(dtype: str, variant: str, causal: bool = True) -> bytes:
    return _profile_attn("B200-SXM-192GB", dtype, variant, causal)


def _profile_attn(gpu: str, dtype: str, variant: str, causal: bool) -> bytes:
    """Single-GPU profile body, run inside a Modal container."""
    import pandas as pd
    import pynvml
    import torch
    from flash_attn import flash_attn_func

    if variant not in ("mha", "gqa", "swa"):
        raise ValueError(f"variant must be mha/gqa/swa, got {variant}")

    pynvml.nvmlInit()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if cvd.startswith("GPU-") or cvd.startswith("MIG-"):
        handle = pynvml.nvmlDeviceGetHandleByUUID(cvd.encode())
    elif cvd:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(cvd))
    else:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    torch_dtype = {
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"unsupported dtype for flash-attn: {dtype}")

    try:
        mem_clk = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        graph_clk = pynvml.nvmlDeviceGetMaxClockInfo(
            handle, pynvml.NVML_CLOCK_GRAPHICS
        )
    except pynvml.NVMLError:
        mem_clk = graph_clk = 0

    power_log = f"/tmp/power_log_{dtype}_{variant}.csv"

    def start_power_log(tag: str, interval_ms: int = 50):
        stop_event = threading.Event()

        def logger():
            with open(power_log, "a") as f:
                while not stop_event.is_set():
                    try:
                        p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except pynvml.NVMLError:
                        p = 0.0
                    f.write(f"{p},{time.time()},{tag}\n")
                    f.flush()
                    time.sleep(interval_ms / 1000.0)

        t = threading.Thread(target=logger)
        t.start()
        return stop_event, t

    def avg_power(tag: str) -> float:
        try:
            with open(power_log) as f:
                vals = [
                    float(line.split(",")[0])
                    for line in f
                    if line.strip() and line.strip().endswith(tag)
                ]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            return 0.0

    def profile_one(
        h: int, d: int, b: int, l: int, h_kv: int, window: int,
    ) -> Tuple[float, float]:
        q = torch.randn(b, l, h, d, device="cuda", dtype=torch_dtype)
        k = torch.randn(b, l, h_kv, d, device="cuda", dtype=torch_dtype)
        v = torch.randn(b, l, h_kv, d, device="cuda", dtype=torch_dtype)
        window_size = (window - 1, 0) if window > 0 else (-1, -1)
        tag = f"v{variant}_h{h}_hkv{h_kv}_d{d}_b{b}_l{l}_w{window}"

        def call():
            return flash_attn_func(
                q, k, v, causal=causal, window_size=window_size,
            )

        # Warmup.
        for _ in range(NUM_WARMUP):
            call()
        torch.cuda.synchronize()

        # Adaptive iter count — keep total measurement near TARGET_WINDOW_S so
        # the power sampler captures multiple ticks even on very fast shapes.
        t0 = time.time()
        call()
        torch.cuda.synchronize()
        single_s = max(time.time() - t0, 1e-6)
        iters = max(MIN_ITER, min(NUM_ITER, int(TARGET_WINDOW_S / single_s)))

        stop_event, thread = start_power_log(tag)
        start = time.time()
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
        end = time.time()
        stop_event.set()
        thread.join()

        time_us = (end - start) / iters * 1e6
        return time_us, avg_power(tag)

    # Enumerate the cells once so we can print a count at the start.
    cells: List[Tuple[int, int, int, int, int, int]] = []
    for h in H:
        if variant == "gqa":
            h_kv_set = sorted({
                h // r for r in H_KV_RATIOS if h % r == 0 and h // r >= 1
            })
        else:
            h_kv_set = [h]
        window_set = WINDOWS if variant == "swa" else [-1]
        for d in D:
            for b in B:
                for l in L:
                    if b * l > MAX_NUM_TOKENS:
                        continue
                    for h_kv in h_kv_set:
                        for w in window_set:
                            if w > 0 and w > l:
                                continue
                            cells.append((h, d, b, l, h_kv, w))

    print(f"[{gpu}/{dtype}/{variant}] {len(cells)} cells to profile")

    data: List[Tuple] = []
    for idx, (h, d, b, l, h_kv, w) in enumerate(cells):
        try:
            t_us, ap = profile_one(h, d, b, l, h_kv, w)
        except Exception as e:
            print(f"[{gpu}/{dtype}/{variant}] skip "
                  f"h={h} d={d} b={b} l={l} h_kv={h_kv} w={w}: {e}")
            continue
        avg_energy = int(t_us) * ap
        w_left = w - 1 if w > 0 else -1
        w_right = 0 if w > 0 else -1
        data.append((
            gpu, dtype, variant, h, h_kv, d, b, l, w_left, w_right,
            int(causal), int(t_us), mem_clk, graph_clk, ap, avg_energy,
        ))
        if (idx + 1) % 100 == 0:
            print(f"[{gpu}/{dtype}/{variant}] {idx + 1}/{len(cells)}")

    pynvml.nvmlShutdown()

    df = pd.DataFrame(data, columns=[
        "gpu", "dtype", "variant", "num_heads", "num_heads_kv", "head_size",
        "batch_size", "seq_len", "window_left", "window_right", "causal",
        "time(us)", "mem_clk_freq", "graph_clk_freq",
        "avg_power(W)", "avg_energy(uJ)",
    ])
    return df.to_csv(index=False).encode()


@app.local_entrypoint()
def main(
    gpu: str = "H100-SXM-80GB,B200-SXM-192GB",
    dtype: str = "bfloat16",
    variant: str = "gqa",
    causal: bool = True,
    shards_dir: str = "flash_attn_shards",
):
    """Run profile on each requested GPU and write one CSV per (gpu, variant).

    --gpu accepts a comma-separated list. Each entry maps to its own Modal
    function (profile_h100 / profile_b200) and runs in its own container.
    """
    os.makedirs(shards_dir, exist_ok=True)

    gpu_list = [g.strip() for g in gpu.split(",") if g.strip()]
    handles = []
    for g in gpu_list:
        if g == "H100-SXM-80GB":
            h = profile_h100.spawn(dtype, variant, causal)
        elif g == "B200-SXM-192GB":
            h = profile_b200.spawn(dtype, variant, causal)
        else:
            raise ValueError(
                f"Unsupported gpu '{g}'. Add a profile_<name> function "
                f"with the matching @app.function(gpu=...) annotation."
            )
        handles.append((g, h))

    for g, h in handles:
        csv_bytes = h.get()
        out_path = os.path.join(shards_dir, f"{g}_{dtype}_{variant}.csv")
        with open(out_path, "wb") as f:
            f.write(csv_bytes)
        print(f"wrote {out_path} ({len(csv_bytes)} bytes)")
