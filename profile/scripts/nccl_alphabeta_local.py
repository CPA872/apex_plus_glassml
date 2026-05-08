"""Standalone NCCL profiler for NVL72-class systems (single box, up to 72
GPUs in one NVLink domain). Self-contained: sweeps every world size from 2
through 72 across 5 collectives at log-spaced message sizes, so a single
run produces all the data apex needs for that GPU type.

Run via torchrun on the NVL72 host:
    torchrun --standalone --nproc_per_node=72 nccl_alphabeta_local.py \\
        --gpu B200-SXM-192GB

Output: nccl_alphabeta_out_nvl72/<gpu>_nvl72_alphabeta.csv

Then convert with profile/scripts/convert_alphabeta_to_apex.py to populate
profile/comm/<gpu>/{all_reduce,alltoall,all_gather,reduce_scatter,sendrecv}.csv.
"""
import argparse
import csv
import os
import time
from typing import List, Tuple


# Same byte-size grid as nccl_alphabeta_modal.py: dense in the 1-64 MB knee
# region, coarse elsewhere.
SIZES_BYTES = [
    1 << 10,                  # 1 KB
    1 << 14,                  # 16 KB
    1 << 16,                  # 64 KB
    1 << 18,                  # 256 KB
    1 << 20,                  # 1 MB
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
    1 << 26,                  # 64 MB
    1 << 27,                  # 128 MB
    1 << 28,                  # 256 MB
    1 << 29,                  # 512 MB
    1 << 30,                  # 1 GB
]

# Full world-size sweep: covers DGX-internal {2, 4, 8} and NVL72 rack-scale
# {16, 32, 64, 72} in one run. Override via --worlds if needed.
DEFAULT_WORLDS = [2, 4, 8, 16, 32, 64, 72]


def _time_one(op: str, group, sub_world: int, n_elems: int) -> float:
    """One timed collective call. n_elems is the total tensor size (S);
    AllGather/ReduceScatter chunk to S/N per rank internally."""
    import torch
    import torch.distributed as dist

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
        buf = torch.ones(n_elems, dtype=torch.bfloat16, device="cuda")
        rank_in_group = dist.get_rank(group)

        def call():
            if rank_in_group == 0:
                dist.send(buf, dst=1, group=group)
            elif rank_in_group == 1:
                dist.recv(buf, src=0, group=group)

    else:
        raise ValueError(f"unknown op: {op}")

    # 2 warmups so NCCL settles algorithm choice.
    call()
    call()
    torch.cuda.synchronize()

    # Adaptive iter count: target ~1 GiB total bytes per measurement.
    bytes_per_iter = n_elems * 2
    n_iter = min(max(10, (1 << 30) // bytes_per_iter), 1 << 13)
    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(n_iter):
        call()
    torch.cuda.synchronize()
    toc = time.time()
    torch.cuda.empty_cache()
    return (toc - tic) / n_iter * 1e6  # microseconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=str, required=True,
        help="GPU label for the CSV (e.g. B200-SXM-192GB)",
    )
    parser.add_argument(
        "--worlds", type=str,
        default=",".join(str(w) for w in DEFAULT_WORLDS),
        help="Comma-separated world sizes to sweep "
             f"(default: {DEFAULT_WORLDS}). Each must be ≤ nproc_per_node.",
    )
    parser.add_argument(
        "--out-dir", type=str, default="nccl_alphabeta_out_nvl72",
        help="Directory to write the CSV (rank 0 only).",
    )
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Squash the unbatched-P2P serialization warning (irrelevant for our
    # one-call-at-a-time SendRecv timing).
    os.environ.setdefault(
        "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false"
    )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    target_worlds = sorted({int(w) for w in args.worlds.split(",") if w.strip()})
    target_worlds = [w for w in target_worlds if 2 <= w <= world_size]
    if not target_worlds:
        if rank == 0:
            print(f"No valid worlds in --worlds for nproc={world_size}")
        dist.destroy_process_group()
        return

    OPS = {
        "allreduce":     target_worlds,
        "alltoall":      target_worlds,
        "allgather":     target_worlds,
        "reducescatter": target_worlds,
        "sendrecv":      [2] if 2 in target_worlds else [],
    }

    rows: List[Tuple] = []

    for sub_world in target_worlds:
        ranks = list(range(sub_world))
        group = dist.new_group(ranks=ranks)
        if rank not in ranks:
            continue
        for op, allowed in OPS.items():
            if sub_world not in allowed:
                continue
            for size_b in SIZES_BYTES:
                n_elems = size_b // 2
                # AllToAll / AllGather / ReduceScatter need n_elems divisible
                # by sub_world. Round DOWN so the actual measured size is
                # slightly smaller than nominal; then record the actual
                # size in the CSV (drift < 2% — invisible in lookup).
                if op in ("alltoall", "allgather", "reducescatter"):
                    n_elems = (n_elems // sub_world) * sub_world
                    if n_elems == 0:
                        continue  # buffer too small for this world
                actual_size_b = n_elems * 2  # bf16
                try:
                    t_us = _time_one(op, group, sub_world, n_elems)
                except Exception as e:
                    if rank == 0:
                        print(
                            f"skip {op} world={sub_world} size={size_b}: {e}"
                        )
                    continue
                if rank == 0:
                    rows.append(
                        (
                            args.gpu, "bfloat16", op, sub_world,
                            actual_size_b, float(t_us),
                        )
                    )
                    print(
                        f"[{args.gpu}/{op}/world={sub_world}] "
                        f"size={actual_size_b:>10}B  time={t_us:8.2f} µs"
                    )

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(
            args.out_dir, f"{args.gpu}_nvl72_alphabeta.csv"
        )
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["gpu", "dtype", "op", "world_size", "size_bytes", "time_us"]
            )
            w.writerows(rows)
        print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
