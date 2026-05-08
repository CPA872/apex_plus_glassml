import argparse
import time
from typing import List, Tuple

import pandas as pd
import ray
import torch

import pynvml
import subprocess
import os
import signal
import threading


NUM_WARMUP = 5
NUM_ITER = 100

# M: Hidden/intermediate dimension up to 64K.
M = [1 << i for i in range(17)]
# K: Hidden/intermediate dimension up to 64K.
K = [1 << i for i in range(17)]
# N: Number of tokens up to 64K
N = list(range(1, 128)) + [128 * i for i in range(1, 513)]


def _build_gemm_fn(m: int, k: int, n: int, dtype: str):
    """Return a no-arg callable that runs one GEMM of the given shape/dtype.

    fp8 takes the e4m3fn _scaled_mm path with bf16 output and unit scales;
    cuBLASLt requires k % 16 == 0 and n % 16 == 0 for that path.
    """
    if dtype in ("half", "fp16"):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        y = torch.randn(k, n, device="cuda", dtype=torch.float16)
        return lambda: torch.matmul(x, y)
    if dtype in ("bf16", "bfloat16"):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        return lambda: torch.matmul(x, y)
    if dtype == "float":
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        y = torch.randn(k, n, device="cuda", dtype=torch.float32)
        return lambda: torch.matmul(x, y)
    if dtype in ("fp8", "fp8_e4m3"):
        if k % 16 or n % 16:
            raise ValueError(
                f"fp8 requires k and n divisible by 16; got m={m} k={k} n={n}"
            )
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        # _scaled_mm needs B in column-major layout; build as (n,k) then .t().
        y = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        ).t()
        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")
        return lambda: torch._scaled_mm(
            x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
        )
    raise ValueError(f"Invalid dtype: {dtype}")


@ray.remote(num_gpus=1)
class GemmProfiler:

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

    def _profile(self, m: int, k: int, n: int, dtype: str) -> Tuple[float, float]:
        gemm_fn = _build_gemm_fn(m, k, n, dtype)

        tag = f"m{m}_k{k}_n{n}"
        power_log_file = f"power_log_{self.idx}.csv"
        stop_event, thread = self._start_power_logging(power_log_file, tag)

        # Warmup
        for _ in range(NUM_WARMUP):
            gemm_fn()
        torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(NUM_ITER):
            gemm_fn()
        torch.cuda.synchronize()
        end = time.time()

        time.sleep(0.1)
        self._stop_power_logging(stop_event, thread)
        time.sleep(0.1)

        avg_power = self._get_avg_power(power_log_file, tag)
        avg_time = (end - start) / NUM_ITER
        return avg_time * 1000 * 1000, avg_power

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
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Exit code: {e.returncode}")
            print(f"Stderr: {e.stderr.strip()}")
            exit(1)

    def _reset_gpu_freq(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-rac"], check=True, capture_output=True, text=True
            )
            print("Clocks reset to default")
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print("Failed to reset clocks")
            print("Exit code:", e.returncode)
            print("Error:", e.stderr.strip())

    def _start_power_logging(self, log_file: str, tag: str, interval_ms: int = 50):
        stop_event = threading.Event()

        def logger():
            with open(log_file, "a") as f:
                while not stop_event.is_set():
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    except pynvml.NVMLError:
                        power = 0.0
                    timestamp = time.time()
                    f.write(f"{power},{timestamp},{tag}\n")
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
                    for line in f
                    if line.strip() and line.strip().endswith(tag)
                ]
            return sum(readings) / len(readings) if readings else 0.0
        except Exception as e:
            print(f"Error reading power log for tag={tag}: {e}")
            return 0.0

    def profile(
        self,
        gpu: str,
        dtype: str,
    ) -> pd.DataFrame:
        data: List[Tuple[str, str, int, int, int, int]] = []
        freq_pairs = self._get_gpu_freq_pairs()
        for mem_clk, graph_clk in freq_pairs:
            print(f"Changing frequency to {mem_clk},{graph_clk} ")

            try:
                self._set_gpu_freq(mem_clk, graph_clk)
                for mi, m in enumerate(M):
                    for ki, k in enumerate(K):
                        for ni, n in enumerate(N):
                            i = mi * len(K) * len(N) + ki * len(N) + ni
                            if i % self.num_gpus != self.idx:
                                continue
                            try:
                                t, avg_power = self._profile(m, k, n, dtype)
                            except (ValueError, RuntimeError) as e:
                                # FP8 alignment (k/n %16) and cuBLAS layout
                                # constraints fail for some small shapes; skip
                                # rather than abort the sweep.
                                print(f"skip dtype={dtype} m={m} k={k} n={n}: {e}")
                                continue
                            avg_energy = int(t) * avg_power
                            data.append(
                                (
                                    gpu,
                                    dtype,
                                    m,
                                    k,
                                    n,
                                    int(t),
                                    mem_clk,
                                    graph_clk,
                                    avg_power,
                                    avg_energy,
                                )
                            )
                        print(f"Finished profiling m={m}, k={k}")
            except Exception as e:
                print(
                    f"Error when profiling at frequency of {mem_clk}, {graph_clk}: {e}"
                )
                continue
        self._reset_gpu_freq()
        pynvml.nvmlShutdown()
        df = pd.DataFrame(
            data,
            columns=[
                "gpu",
                "dtype",
                "m",
                "k",
                "n",
                "time(us)",
                "mem_clk_freq",
                "graph_clk_freq",
                "avg_power(W)",
                "avg_energy(uJ)",
            ],
        )
        return df


def main(gpu: str, num_gpus: int, dtype: str):
    for filename in os.listdir():
        if filename.startswith("power_log_") and filename.endswith(".csv"):
            os.remove(filename)

    profilers = [GemmProfiler.remote(i, num_gpus) for i in range(num_gpus)]
    profiled_data = []
    for i in range(num_gpus):
        profiler = profilers[i]
        data = profiler.profile.remote(gpu, dtype)
        profiled_data.append(data)
    profiled_data = ray.get(profiled_data)
    df = pd.concat(profiled_data, ignore_index=True)
    df = df.sort_values(
        by=["gpu", "dtype", "m", "k", "n", "mem_clk_freq", "graph_clk_freq"]
    )
    df.to_csv("gemm.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=str, required=True, choices=["V100-PCIE-16GB", "H100-SXM-80GB", "RTX-PRO6000-BLACKWELL", "B200-SXM-192GB"]
    )
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        choices=["half", "fp16", "bfloat16", "bf16", "float", "fp8", "fp8_e4m3"],
    )
    args = parser.parse_args()

    print(args)
    main(args.gpu, args.num_gpus, args.dtype)
