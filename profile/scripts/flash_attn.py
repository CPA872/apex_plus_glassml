"""
FlashAttention (v2/v3) profiler — MHA, GQA, and SWA variants.

Follows the same Ray + NVML pattern as mha.py/gemm.py. Replaces the xFormers
CUTLASS backend with flash_attn_func, which is what modern serving stacks
actually use on Hopper/Blackwell.

Variants (select via --variant):
    mha : standard multi-head attention (num_heads_kv = num_heads_q)
    gqa : grouped-query attention; sweeps num_heads_kv = num_heads / {2, 4, 8}
    swa : sliding-window attention; sweeps window_size in {128, 512, 2048, 4096}

Output: flash_attn_<variant>.csv in the current directory. Columns match
mha.py conventions, extended with num_heads_kv, window_left, window_right,
causal, and variant so downstream tooling can filter per kind.
"""
import argparse
import os
import threading
import time

import pandas as pd
import pynvml
import ray
import subprocess
import torch
from flash_attn import flash_attn_func
from torch.profiler import profile, ProfilerActivity

NUM_WARMUP = 5

# H: Number of attention heads up to 128.
H = [1, 2, 4, 8, 16, 24, 32, 40, 48, 52, 64, 96, 128]
# D: Head dimension (FA2-supported values).
D = [64, 96, 128, 192, 256]
# B: Batch size up to 4096.
B = list(range(1, 32)) + [1 << i for i in range(5, 13)]
# L: Sequence length up to 16K.
L = [16 * i for i in range(1, 1025)]
# B * L <= MAX_NUM_TOKENS.
MAX_NUM_TOKENS = 16 * 4096

# GQA: num_heads_q / num_heads_kv ratios to sweep.
H_KV_RATIOS = [2, 4, 8]
# SWA: window sizes to sweep (causal-only: window is (W-1, 0)).
WINDOWS = [128, 512, 2048, 4096]


@ray.remote(num_gpus=1)
class FlashAttnProfiler:

    def __init__(self, idx: int, num_gpus: int):
        self.idx = idx
        self.num_gpus = num_gpus
        pynvml.nvmlInit()
        # Ray sets CUDA_VISIBLE_DEVICES per actor; NVML indices are physical, so
        # resolve via the visible device's UUID/index instead of the actor index.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if cvd.startswith("GPU-") or cvd.startswith("MIG-"):
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(cvd.encode())
        elif cvd:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(cvd))
        else:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def _get_gpu_freq_pairs(self):
        try:
            memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
            max_mem_clk = max(memory_clocks)
            graphics_clks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                self.handle, max_mem_clk
            )
            max_graph_clk = max(graphics_clks)
        except pynvml.NVMLError as e:
            print(f"Error getting supported GPU clocks from NVML: {e}")
            exit(1)
        return [(max_mem_clk, max_graph_clk)]

    def _set_gpu_freq(self, mem_clk, graph_clk):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-ac", f"{mem_clk},{graph_clk}"],
                check=True, capture_output=True, text=True,
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Exit code: {e.returncode}")
            print(f"Stderr: {e.stderr.strip()}")
            exit(1)

    def _reset_gpu_freq(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-rac"], check=True, capture_output=True, text=True,
            )
            print("Clocks reset to default")
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print("Failed to reset clocks")
            print(f"Exit code: {e.returncode}")
            print(f"Stderr: {e.stderr.strip()}")

    def _start_power_logging(self, log_file: str, tag: str, interval_ms: int = 50):
        stop_event = threading.Event()

        def logger():
            with open(log_file, "a") as f:
                while not stop_event.is_set():
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    except pynvml.NVMLError:
                        power = 0.0
                    f.write(f"{power},{time.time()},{tag}\n")
                    f.flush()
                    time.sleep(interval_ms / 1000.0)

        t = threading.Thread(target=logger)
        t.start()
        return stop_event, t

    def _stop_power_logging(self, stop_event, thread):
        stop_event.set()
        thread.join()

    def _get_avg_power(self, log_file: str, tag: str) -> float:
        try:
            with open(log_file) as f:
                readings = [
                    float(line.split(",")[0])
                    for line in f if line.strip() and line.strip().endswith(tag)
                ]
            return sum(readings) / len(readings) if readings else 0.0
        except Exception as e:
            print(f"Error reading power log for tag={tag}: {e}")
            return 0.0

    def _profile(self, h: int, d: int, b: int, l: int, h_kv: int,
                 window: int, causal: bool, dtype_str: str, variant: str):
        if dtype_str == "half":
            dtype = torch.float16
        elif dtype_str == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid dtype: {dtype_str}")

        q = torch.randn(b, l, h, d, device="cuda", dtype=dtype)
        k = torch.randn(b, l, h_kv, d, device="cuda", dtype=dtype)
        v = torch.randn(b, l, h_kv, d, device="cuda", dtype=dtype)
        window_size = (window - 1, 0) if window > 0 else (-1, -1)

        tag = f"{variant}_h{h}_hkv{h_kv}_d{d}_b{b}_l{l}_w{window}"
        power_log = f"flash_attn_{variant}_power_log_{self.idx}.csv"
        stop_event, thread = self._start_power_logging(power_log, tag)

        for _ in range(NUM_WARMUP):
            flash_attn_func(q, k, v, causal=causal, window_size=window_size)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            flash_attn_func(q, k, v, causal=causal, window_size=window_size)
        time.sleep(0.05)
        self._stop_power_logging(stop_event, thread)
        time.sleep(0.05)

        avg_power = self._get_avg_power(power_log, tag)
        stats = prof.key_averages()
        device_time_us = sum(s.device_time for s in stats)
        return device_time_us, avg_power

    def profile(self, gpu: str, dtype: str, variant: str, causal: bool) -> pd.DataFrame:
        data = []
        freq_pairs = self._get_gpu_freq_pairs()
        for mem_clk, graph_clk in freq_pairs:
            print(f"Changing frequency to {mem_clk},{graph_clk}")
            try:
                self._set_gpu_freq(mem_clk, graph_clk)
                shape_idx = 0
                for h in H:
                    if variant == "gqa":
                        h_kv_set = [h // r for r in H_KV_RATIOS if h % r == 0 and h // r >= 1]
                        h_kv_set = sorted(set(h_kv_set))
                    else:
                        h_kv_set = [h]
                    window_set = WINDOWS if variant == "swa" else [-1]

                    for d in D:
                        for b in B:
                            for l in L:
                                if b * l > MAX_NUM_TOKENS:
                                    continue
                                for h_kv in h_kv_set:
                                    for window in window_set:
                                        if window > 0 and window > l:
                                            continue
                                        if shape_idx % self.num_gpus != self.idx:
                                            shape_idx += 1
                                            continue
                                        shape_idx += 1
                                        try:
                                            t, avg_power = self._profile(
                                                h, d, b, l, h_kv, window,
                                                causal, dtype, variant,
                                            )
                                        except (RuntimeError, ValueError) as e:
                                            print(f"Skip h={h} d={d} b={b} l={l} "
                                                  f"h_kv={h_kv} w={window}: {e}")
                                            continue
                                        avg_energy = int(t) * avg_power
                                        w_left = window - 1 if window > 0 else -1
                                        w_right = 0 if window > 0 else -1
                                        data.append((
                                            gpu, dtype, variant, h, h_kv, d,
                                            b, l, w_left, w_right, int(causal),
                                            int(t), mem_clk, graph_clk,
                                            avg_power, avg_energy,
                                        ))
            except Exception as e:
                print(f"Error at freq {mem_clk},{graph_clk}: {e}")
                continue
        self._reset_gpu_freq()
        pynvml.nvmlShutdown()
        df = pd.DataFrame(data, columns=[
            "gpu", "dtype", "variant", "num_heads", "num_heads_kv", "head_size",
            "batch_size", "seq_len", "window_left", "window_right", "causal",
            "time(us)", "mem_clk_freq", "graph_clk_freq",
            "avg_power(W)", "avg_energy(uJ)",
        ])
        return df


def main(gpu: str, num_gpus: int, dtype: str, variant: str, causal: bool):
    for filename in os.listdir():
        if filename.startswith(f"flash_attn_{variant}_power_log_") and filename.endswith(".csv"):
            os.remove(filename)

    profilers = [FlashAttnProfiler.remote(i, num_gpus) for i in range(num_gpus)]
    results = ray.get([p.profile.remote(gpu, dtype, variant, causal) for p in profilers])
    df = pd.concat(results, ignore_index=True)
    df = df.sort_values(by=["gpu", "dtype", "variant", "num_heads", "num_heads_kv",
                            "head_size", "batch_size", "seq_len",
                            "window_left", "causal"])
    df.to_csv(f"flash_attn_{variant}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True,
                        choices=["V100-PCIE-16GB", "H100-SXM-80GB",
                                 "RTX-PRO6000-BLACKWELL", "B200-SXM-192GB"])
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["half", "bfloat16"])
    parser.add_argument("--variant", type=str, required=True,
                        choices=["mha", "gqa", "swa"])
    parser.add_argument("--no-causal", action="store_true",
                        help="Disable causal mask (for bidirectional attention).")
    args = parser.parse_args()

    print(args)
    main(args.gpu, args.num_gpus, args.dtype, args.variant, not args.no_causal)
