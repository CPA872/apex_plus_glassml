"""Modal port of gemm.py — profiles torch.matmul across (M, K, N) on remote GPUs.

Fan-out is by dtype: one Modal container per dtype, each sweeping the full
M x K x N grid for that dtype and writing its own CSV (gemm_shards/<dtype>.csv).
With the default --dtypes "half,bfloat16,float,fp8" that's 4 parallel
containers. Per-dtype files mean partial results land on disk even if one
dtype fails.

GPU clock pinning (nvidia-smi -ac) is dropped because Modal containers are
not privileged; the columns are kept for schema parity and filled with the
device's reported max clocks.
"""
import io
import os
import threading
import time
from typing import List, Tuple

import modal

from flash_attn_image import flash_attn_image

app = modal.App("gemm-profile")

# Ship flash_attn_image.py into the container so the import at the top of
# this file resolves remotely too (Modal no longer auto-mounts siblings).
image = flash_attn_image.add_local_python_source("flash_attn_image")

MODAL_GPU = os.environ.get("MODAL_GPU", "B200")

NUM_WARMUP = 5
NUM_ITER = 100        # cap on iters per shape
MIN_ITER = 5          # floor when shapes take seconds each
TARGET_WINDOW_S = 0.2 # aim for ~200ms total measurement window

M = [1 << i for i in range(18)]  # 1, 2, 4, ..., 131072
K = [1 << i for i in range(18)]  # 1, 2, 4, ..., 131072
# Small N (1..127) for kernel-launch regime; dense (step 128) up to 16K where
# tile selection still varies; sparse (step 1024) above 16K where shapes are
# compute-bound and time is linear in N (piecewise-linear interp loses ~0).
N = (
    list(range(1, 128))                    # 1..127 (127 vals)
    + [128 * i for i in range(1, 128)]     # 128..16256, step 128 (127 vals)
    + [1024 * i for i in range(16, 129)]   # 16384..131072, step 1024 (113 vals)
)  # 367 values total

# Shapes where every dim is at least this size are well-approximated by the
# analytic roofline 2*m*k*n / peak_flops, so skip them — measuring them just
# burns GPU-hours to confirm the obvious.
SKIP_MIN_DIM = 8192


@app.function(image=image, gpu=MODAL_GPU, timeout=86400)
def profile_shard(gpu: str, dtype: str) -> bytes:
    import pandas as pd
    import pynvml
    import torch

    pynvml.nvmlInit()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if cvd.startswith("GPU-") or cvd.startswith("MIG-"):
        handle = pynvml.nvmlDeviceGetHandleByUUID(cvd.encode())
    elif cvd:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(cvd))
    else:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # torch_dtype is only used by the standard torch.matmul path. The fp8
    # branch builds float8_e4m3fn tensors directly inside _build_gemm_fn.
    torch_dtype = {
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float": torch.float32,
        "fp8": None,
        "fp8_e4m3": None,
    }.get(dtype)
    if dtype not in {"half", "fp16", "bfloat16", "bf16", "float", "fp8", "fp8_e4m3"}:
        raise ValueError(f"Invalid dtype: {dtype}")

    def build_gemm_fn(m: int, k: int, n: int):
        if dtype in ("fp8", "fp8_e4m3"):
            if k % 16 or n % 16:
                raise ValueError(
                    f"fp8 requires k and n divisible by 16; got m={m} k={k} n={n}"
                )
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            y = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ).t()
            scale_a = torch.tensor(1.0, device="cuda")
            scale_b = torch.tensor(1.0, device="cuda")
            return lambda: torch._scaled_mm(
                x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
            )
        x = torch.randn(m, k, device="cuda", dtype=torch_dtype)
        y = torch.randn(k, n, device="cuda", dtype=torch_dtype)
        return lambda: torch.matmul(x, y)

    try:
        mem_clk = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        graph_clk = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    except pynvml.NVMLError:
        mem_clk = graph_clk = 0

    power_log = f"/tmp/power_log_{dtype}.csv"

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

    def profile_one(m: int, k: int, n: int) -> Tuple[float, float]:
        gemm_fn = build_gemm_fn(m, k, n)
        tag = f"m{m}_k{k}_n{n}"

        for _ in range(NUM_WARMUP):
            gemm_fn()
        torch.cuda.synchronize()

        # One untimed iter to size the measurement loop. Goal: keep total
        # measurement time ~TARGET_WINDOW_S so power sampling has multiple
        # ticks to land inside the active window, and large shapes don't
        # eat minutes per row at NUM_ITER=100.
        t0 = time.time()
        gemm_fn()
        torch.cuda.synchronize()
        single_s = max(time.time() - t0, 1e-6)
        iters = max(MIN_ITER, min(NUM_ITER, int(TARGET_WINDOW_S / single_s)))

        stop_event, thread = start_power_log(tag)
        start = time.time()
        for _ in range(iters):
            gemm_fn()
        torch.cuda.synchronize()
        end = time.time()
        stop_event.set()
        thread.join()

        return (end - start) / iters * 1e6, avg_power(tag)

    is_fp8 = dtype in ("fp8", "fp8_e4m3")
    n_total = sum(
        1 for m in M for k in K for n in N
        if min(m, k, n) < SKIP_MIN_DIM
        and not (is_fp8 and (k % 16 or n % 16))
    )
    extra = " and k,n divisible by 16" if is_fp8 else ""
    print(
        f"[{dtype}] {n_total} shapes total "
        f"(min(m,k,n) < {SKIP_MIN_DIM}{extra})"
    )

    data: List[Tuple] = []
    for m in M:
        for k in K:
            if is_fp8 and k % 16:
                continue
            for n in N:
                if min(m, k, n) >= SKIP_MIN_DIM:
                    continue
                if is_fp8 and n % 16:
                    continue
                try:
                    t_us, ap = profile_one(m, k, n)
                except Exception as e:
                    print(f"[{dtype}] skip m={m} k={k} n={n}: {e}")
                    continue
                data.append((
                    gpu, dtype, m, k, n, int(t_us),
                    mem_clk, graph_clk, ap, int(t_us) * ap,
                ))
            print(f"[{dtype}] finished m={m} k={k}")

    pynvml.nvmlShutdown()

    df = pd.DataFrame(data, columns=[
        "gpu", "dtype", "m", "k", "n", "time(us)",
        "mem_clk_freq", "graph_clk_freq", "avg_power(W)", "avg_energy(uJ)",
    ])
    return df.to_csv(index=False).encode()


@app.local_entrypoint()
def main(
    gpu: str = "B200-SXM-192GB",
    dtypes: str = "half,bfloat16,float,fp8",
    shards_dir: str = "gemm_shards",
):
    """Write one raw CSV per dtype (10-column schema) to <shards_dir>/.

    Fan-out is by dtype — one Modal container per dtype, all running in
    parallel. With the default 4 dtypes that's 4 containers; each sweeps
    the full M x K x N grid and writes <shards_dir>/<dtype>.csv.

    Run aggregate_gemm.py afterward to project to the 6-column simulator
    format and place files under apex_plus/profile/comp/<gpu>/.

    fp8 quietly skips shapes where k%16 or n%16 != 0 (torch._scaled_mm
    alignment requirement); aggregate_gemm.py just sees fewer fp8 rows.
    """
    import pathlib
    import pandas as pd

    dtype_list = [d.strip() for d in dtypes.split(",") if d.strip()]
    out_dir = pathlib.Path(shards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [(gpu, dt) for dt in dtype_list]
    print(
        f"Launching {len(args)} containers on {MODAL_GPU} for gpu={gpu} "
        f"dtypes={dtype_list}"
    )
    print(f"Writing shard CSVs to {out_dir.resolve()}/")

    for arg, result in zip(
        args, profile_shard.starmap(args, return_exceptions=True)
    ):
        dt = arg[1]
        tag = dt
        if isinstance(result, Exception):
            print(f"[{tag}] FAILED: {result}")
            continue
        path = out_dir / f"{tag}.csv"
        df = pd.read_csv(io.BytesIO(result))
        df.to_csv(path, index=False)
        print(f"[{tag}] wrote {len(df)} rows to {path}")
