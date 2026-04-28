"""
FlashMLA profiler — DeepSeek Multi-head Latent Attention (decode mode).

Profiles the flash_mla_with_kvcache kernel against a sweep of
(batch_size, cache_seqlen, num_heads_q). Head dims are fixed to the MLA
convention: d_k = 192 (128 nope + 64 rope), d_v = 128, h_kv = 1 (latent).

Requires deepseek-ai/FlashMLA installed and a Hopper+ GPU. If the import
fails the script errors out with install instructions.

Output: flash_mla.csv in the current directory.
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
from torch.profiler import profile, ProfilerActivity

try:
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache
except ImportError as e:
    raise SystemExit(
        "flash_mla not installed. Install with:\n"
        "  uv pip install git+https://github.com/deepseek-ai/FlashMLA.git\n"
        "Requires a Hopper (sm_90) or Blackwell (sm_100) GPU.\n"
        f"Original error: {e}"
    )

NUM_WARMUP = 5

# MLA architectural constants (DeepSeek-V3 / Kimi-K2).
D_K = 192   # Q/K head dim = qk_nope_head_dim (128) + qk_rope_head_dim (64)
D_V = 128   # V head dim
H_KV = 1    # MLA uses a single latent head in the absorbed form
S_Q = 1     # decode: one new token per request

# Sweep axes.
H_Q = [64, 128]                                      # Kimi=64, DSv3=128
B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CACHE_SEQLEN = [512, 1024, 2048, 4096, 8192, 16384, 32768]
BLOCK_SIZE = [64]


@ray.remote(num_gpus=1)
class FlashMLAProfiler:

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

    def _profile(self, b: int, cache_seqlen: int, h_q: int,
                 block_size: int, dtype_str: str):
        if dtype_str == "half":
            dtype = torch.float16
        elif dtype_str == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid dtype: {dtype_str}")

        num_pages_per_req = (cache_seqlen + block_size - 1) // block_size
        total_pages = b * num_pages_per_req

        q = torch.randn(b, S_Q, h_q, D_K, device="cuda", dtype=dtype)
        kvcache = torch.randn(total_pages, block_size, H_KV, D_K,
                              device="cuda", dtype=dtype)
        block_table = torch.arange(
            total_pages, device="cuda", dtype=torch.int32
        ).reshape(b, num_pages_per_req)
        cache_seqlens = torch.full((b,), cache_seqlen,
                                   device="cuda", dtype=torch.int32)

        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens, S_Q * h_q // H_KV, H_KV,
        )

        tag = f"b{b}_cl{cache_seqlen}_h{h_q}_bs{block_size}"
        power_log = f"flash_mla_power_log_{self.idx}.csv"
        stop_event, thread = self._start_power_logging(power_log, tag)

        for _ in range(NUM_WARMUP):
            flash_mla_with_kvcache(
                q, kvcache, block_table, cache_seqlens, D_V,
                tile_scheduler_metadata, num_splits, causal=True,
            )
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            flash_mla_with_kvcache(
                q, kvcache, block_table, cache_seqlens, D_V,
                tile_scheduler_metadata, num_splits, causal=True,
            )
        time.sleep(0.05)
        self._stop_power_logging(stop_event, thread)
        time.sleep(0.05)

        avg_power = self._get_avg_power(power_log, tag)
        stats = prof.key_averages()
        device_time_us = sum(s.device_time for s in stats)

        bytes_elem = 2
        bytes_read = b * cache_seqlen * H_KV * D_K * bytes_elem
        bandwidth_gbs = bytes_read / (device_time_us * 1e-6) / 1e9 if device_time_us > 0 else 0.0

        return device_time_us, avg_power, bandwidth_gbs

    def profile(self, gpu: str, dtype: str) -> pd.DataFrame:
        data = []
        freq_pairs = self._get_gpu_freq_pairs()
        for mem_clk, graph_clk in freq_pairs:
            print(f"Changing frequency to {mem_clk},{graph_clk}")
            try:
                self._set_gpu_freq(mem_clk, graph_clk)
                shape_idx = 0
                for h_q in H_Q:
                    for b in B:
                        for cache_seqlen in CACHE_SEQLEN:
                            for block_size in BLOCK_SIZE:
                                if shape_idx % self.num_gpus != self.idx:
                                    shape_idx += 1
                                    continue
                                shape_idx += 1
                                try:
                                    t, avg_power, bw = self._profile(
                                        b, cache_seqlen, h_q, block_size, dtype,
                                    )
                                except (RuntimeError, ValueError) as e:
                                    print(f"Skip b={b} cl={cache_seqlen} h={h_q} "
                                          f"bs={block_size}: {e}")
                                    continue
                                avg_energy = int(t) * avg_power
                                data.append((
                                    gpu, dtype, b, cache_seqlen,
                                    h_q, H_KV, D_K, D_V, block_size,
                                    int(t), bw, mem_clk, graph_clk,
                                    avg_power, avg_energy,
                                ))
            except Exception as e:
                print(f"Error at freq {mem_clk},{graph_clk}: {e}")
                continue
        self._reset_gpu_freq()
        pynvml.nvmlShutdown()
        df = pd.DataFrame(data, columns=[
            "gpu", "dtype", "batch_size", "cache_seqlen",
            "num_heads_q", "num_heads_kv", "d_k", "d_v", "block_size",
            "time(us)", "bandwidth(GB/s)", "mem_clk_freq", "graph_clk_freq",
            "avg_power(W)", "avg_energy(uJ)",
        ])
        return df


def main(gpu: str, num_gpus: int, dtype: str):
    for filename in os.listdir():
        if filename.startswith("flash_mla_power_log_") and filename.endswith(".csv"):
            os.remove(filename)

    profilers = [FlashMLAProfiler.remote(i, num_gpus) for i in range(num_gpus)]
    results = ray.get([p.profile.remote(gpu, dtype) for p in profilers])
    df = pd.concat(results, ignore_index=True)
    df = df.sort_values(by=["gpu", "dtype", "num_heads_q", "batch_size",
                            "cache_seqlen", "block_size"])
    df.to_csv("flash_mla.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True,
                        choices=["V100-PCIE-16GB", "H100-SXM-80GB",
                                 "RTX-PRO6000-BLACKWELL", "B200-SXM-192GB"])
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["half", "bfloat16"])
    args = parser.parse_args()

    print(args)
    main(args.gpu, args.num_gpus, args.dtype)
