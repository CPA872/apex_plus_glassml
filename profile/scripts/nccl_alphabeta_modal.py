"""Modal launcher: profile NCCL collectives across log-spaced message sizes
to expose the latency (α) and bandwidth (β) regimes of the interconnect.

Single 8-GPU container per Modal invocation (intra-node NVLink only). Sweeps
AllReduce, AllToAll, AllGather, ReduceScatter, and SendRecv on world subgroups
of {2, 4, 8} (SendRecv is pair-wise only, so only world=2). The selected
message sizes straddle both regimes:

    1 KB → 256 KB :  latency-bound  (T ≈ α, plateau)
    1 MB → 16 MB  :  knee
    64 MB → 1 GB  :  bandwidth-bound (T ≈ S/β, slope=1 in log-log)

Each row in the output CSV is one timed run; downstream `plot_alphabeta.py`
fits T = α + S/β per (gpu, op, world) and produces log-log plots.

Usage:
    cd apex_plus/profile/scripts
    modal run nccl_alphabeta_modal.py --gpu "H100-SXM-80GB,B200-SXM-192GB"

Cost: ~$3-5 per H100 run (~5-10 min wall), ~$5-7 per B200.
"""
import os
import time
from typing import List, Tuple

import modal

app = modal.App("nccl-alphabeta")

# Use cu130 + torch 2.9 so the bundled NCCL has Blackwell (sm_100) kernels.
# PyTorch ≤2.6 / cu124 was built pre-B200 and crashes on certain large-payload
# AllToAll variants with "no kernel image is available" on sm_100.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("libibverbs-dev")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0;10.0;12.0+PTX"})
    .pip_install(
        "torch==2.9.*",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("pandas")
)

# Byte sizes spanning both regimes. Coarse in latency-bound region (small),
# **dense in the 1-64 MB knee region** to capture NCCL algorithm switches,
# coarse again in the bandwidth-bound tail.
SIZES_BYTES = [
    1 << 10,                  # 1 KB        ← deep latency
    1 << 14,                  # 16 KB
    1 << 16,                  # 64 KB
    1 << 18,                  # 256 KB
    1 << 20,                  # 1 MB        ── knee region begins ──
    int(1.5 * (1 << 20)),     # 1.5 MB
    1 << 21,                  # 2 MB
    int(3 * (1 << 20)),       # 3 MB
    1 << 22,                  # 4 MB
    int(6 * (1 << 20)),       # 6 MB
    1 << 23,                  # 8 MB
    int(12 * (1 << 20)),      # 12 MB
    1 << 24,                  # 16 MB
    int(24 * (1 << 20)),      # 24 MB
    1 << 25,                  # 32 MB
    1 << 26,                  # 64 MB       ── bandwidth saturation ──
    1 << 27,                  # 128 MB
    1 << 28,                  # 256 MB
    1 << 29,                  # 512 MB
    1 << 30,                  # 1 GB        ← deep bandwidth
]
WORLD_SIZES_INTRANODE = [2, 4, 8]


def _worker(rank: int, world_size: int, gpu_label: str, out_path: str):
    """One process per rank. Rank 0 writes the CSV when done."""
    import csv

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    # Squash the unbatched-P2P serialization warning. We only run one P2P at
    # a time on a 2-rank group, so the warning's concern doesn't apply.
    os.environ.setdefault(
        "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false"
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{rank}"),
    )

    rows: List[Tuple] = []

    def time_one(op: str, group, sub_world: int, n_elems: int) -> float:
        """Time one collective. n_elems = total tensor size in elements; for
        AllGather and ReduceScatter the per-rank chunk is n_elems // sub_world."""
        if op == "allreduce":
            buf = torch.ones(n_elems, dtype=torch.bfloat16, device="cuda")
            def call():
                dist.all_reduce(buf, group=group)

        elif op == "alltoall":
            buf = torch.ones(n_elems, dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(buf)
            def call():
                in_c = list(buf.chunk(sub_world))
                out_c = list(out.chunk(sub_world))
                dist.all_to_all(out_c, in_c, group=group)

        elif op == "allgather":
            chunk = n_elems // sub_world
            in_buf = torch.ones(chunk, dtype=torch.bfloat16, device="cuda")
            out_buf = torch.empty(n_elems, dtype=torch.bfloat16, device="cuda")
            def call():
                dist.all_gather_into_tensor(out_buf, in_buf, group=group)

        elif op == "reducescatter":
            chunk = n_elems // sub_world
            in_buf = torch.ones(n_elems, dtype=torch.bfloat16, device="cuda")
            out_buf = torch.empty(chunk, dtype=torch.bfloat16, device="cuda")
            def call():
                dist.reduce_scatter_tensor(out_buf, in_buf, group=group)

        elif op == "sendrecv":
            # Rank 0 sends, rank 1 receives — pair-wise only. Other ranks
            # idle for the duration; we time on rank 0.
            buf = torch.ones(n_elems, dtype=torch.bfloat16, device="cuda")
            rank_in_group = dist.get_rank(group)
            def call():
                if rank_in_group == 0:
                    dist.send(buf, dst=1, group=group)
                elif rank_in_group == 1:
                    dist.recv(buf, src=0, group=group)
        else:
            raise ValueError(f"unknown op: {op}")

        # 2 warmups so NCCL settles its algorithm choice.
        call()
        call()
        torch.cuda.synchronize()

        # Adaptive iter count: target ~1 GiB total bytes per measurement so
        # power sampling and clock noise average out, capped at 8192 iters.
        bytes_per_iter = n_elems * 2  # bf16
        n_iter = min(max(10, (1 << 30) // bytes_per_iter), 1 << 13)

        torch.cuda.synchronize()
        tic = time.time()
        for _ in range(n_iter):
            call()
        torch.cuda.synchronize()
        toc = time.time()
        torch.cuda.empty_cache()
        return (toc - tic) / n_iter * 1e6  # microseconds

    # Per-collective allowed world sizes. sendrecv is pair-wise only;
    # the rest sweep all intra-node worlds.
    OPS = {
        "allreduce":     WORLD_SIZES_INTRANODE,
        "alltoall":      WORLD_SIZES_INTRANODE,
        "allgather":     WORLD_SIZES_INTRANODE,
        "reducescatter": WORLD_SIZES_INTRANODE,
        "sendrecv":      [2],
    }

    for sub_world in WORLD_SIZES_INTRANODE:
        if sub_world > world_size:
            continue
        ranks = list(range(sub_world))
        group = dist.new_group(ranks=ranks)
        if rank not in ranks:
            continue
        for op, allowed_worlds in OPS.items():
            if sub_world not in allowed_worlds:
                continue
            for size_b in SIZES_BYTES:
                n_elems = size_b // 2  # bf16 = 2 B/element
                # AllToAll, AllGather, ReduceScatter need divisibility by world.
                if op in ("alltoall", "allgather", "reducescatter"):
                    if n_elems % sub_world != 0 or n_elems // sub_world == 0:
                        continue
                try:
                    t_us = time_one(op, group, sub_world, n_elems)
                except Exception as e:
                    if rank == 0:
                        print(f"skip {op} world={sub_world} size={size_b}: {e}")
                    continue
                if rank == 0:
                    rows.append(
                        (gpu_label, "bfloat16", op, sub_world, size_b, float(t_us))
                    )
                    print(
                        f"[{gpu_label}/{op}/world={sub_world}] "
                        f"size={size_b:>10}B  time={t_us:8.2f} µs"
                    )

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gpu", "dtype", "op", "world_size", "size_bytes", "time_us"])
            w.writerows(rows)


def _run_profile(gpu_label: str) -> bytes:
    """Spawn 8 NCCL ranks via mp.spawn, return rank-0's CSV bytes."""
    import torch.multiprocessing as mp

    out_path = "/tmp/nccl_alphabeta.csv"
    if os.path.exists(out_path):
        os.remove(out_path)

    mp.spawn(_worker, args=(8, gpu_label, out_path), nprocs=8, join=True)

    with open(out_path, "rb") as f:
        return f.read()


@app.function(image=image, gpu="H100:8", timeout=3600)
def profile_h100() -> bytes:
    return _run_profile("H100-SXM-80GB")


@app.function(image=image, gpu="B200:8", timeout=3600)
def profile_b200() -> bytes:
    return _run_profile("B200-SXM-192GB")


@app.local_entrypoint()
def main(
    gpu: str = "H100-SXM-80GB,B200-SXM-192GB",
    out_dir: str = "nccl_alphabeta_out",
):
    """Run profile on each requested GPU, write one CSV per GPU."""
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
            raise ValueError(
                f"Unsupported gpu '{g}'. Add a profile_<name> function "
                f"with a matching @app.function(gpu=...) annotation."
            )
        handles.append((g, h))

    for g, h in handles:
        csv_bytes = h.get()
        path = out / f"{g}_alphabeta.csv"
        path.write_bytes(csv_bytes)
        print(f"wrote {path} ({len(csv_bytes)} bytes)")
