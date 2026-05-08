"""Modal port of comm.py — smoke-test NCCL collectives across 2 H100 nodes.

Mirrors the AllReduce + AllToAll paths from comm.py but uses
torch.distributed (NCCL) instead of CuPy + Ray, since Modal natively
provisions a clustered, RDMA-capable network for us.

Topology: 2 containers x 1 H100 each (world_size=2). Cross-container
traffic goes over Modal's RDMA fabric, so this is the IB-equivalent path.
GPU clock pinning is dropped (Modal containers aren't privileged).
"""
import io
import os
import time
from typing import List, Tuple

import modal
import modal.experimental

app = modal.App("comm-profile")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "libibverbs-dev")
    .pip_install(
        "torch==2.5.*",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("pandas", "numpy", "nvidia-ml-py")
)

NUM_NODES = 2
GPUS_PER_NODE = 1
GPU = "H100"
WORLD_SIZE = NUM_NODES * GPUS_PER_NODE

KB = 1024
SIZE_EXPS = list(range(8, 21))  # 1<<8 .. 1<<20 elements (smoke range)


@app.function(image=image, gpu=GPU, timeout=3600)
@modal.experimental.clustered(size=NUM_NODES, rdma=True)
def profile_node():
    import torch
    import torch.distributed as dist

    info = modal.experimental.get_cluster_info()
    rank = info.rank
    world_size = len(info.container_ips)
    master_addr = info.container_ips[0]

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[rank {rank}/{world_size}] init done, master={master_addr}")

    dtype = torch.float16
    bytes_per_el = 2

    def bench(op_fn, in_buf, out_buf):
        op_fn(in_buf, out_buf)
        op_fn(in_buf, out_buf)
        torch.cuda.synchronize()
        n_elems = in_buf.numel()
        N = min(max(10, (1 << 30) // (n_elems * bytes_per_el)), 1 << 13)
        torch.cuda.synchronize()
        tic = time.time()
        for _ in range(N):
            op_fn(in_buf, out_buf)
        torch.cuda.synchronize()
        toc = time.time()
        return (toc - tic) / N

    def all_reduce_op(in_b, out_b):
        out_b.copy_(in_b)
        dist.all_reduce(out_b)

    def all_to_all_op(in_b, out_b):
        in_chunks = list(in_b.chunk(world_size))
        out_chunks = list(out_b.chunk(world_size))
        dist.all_to_all(out_chunks, in_chunks)

    ar_rows: List[Tuple] = []
    a2a_rows: List[Tuple] = []
    header = ["gpu", "num_nodes", "num_gpus_per_node", "dtype",
              "size(kb)", "time(us)", "gpu_freq"]

    for i in SIZE_EXPS:
        size_elems = 1 << i
        in_buf = torch.ones(size_elems, dtype=dtype, device="cuda")
        out_buf = torch.ones(size_elems, dtype=dtype, device="cuda")
        size_kb = size_elems * bytes_per_el // KB

        try:
            t = bench(all_reduce_op, in_buf, out_buf) * 1e6
            ar_rows.append(("H100-SXM-80GB", NUM_NODES, GPUS_PER_NODE,
                            "half", size_kb, int(t), 0))
            if rank == 0:
                print(f"AllReduce  size={size_kb}KB  time={int(t)}us")
        except Exception as e:
            print(f"[rank {rank}] AllReduce {size_elems} failed: {e}")

        try:
            t = bench(all_to_all_op, in_buf, out_buf) * 1e6
            a2a_rows.append(("H100-SXM-80GB", NUM_NODES, GPUS_PER_NODE,
                             "half", size_kb, int(t), 0))
            if rank == 0:
                print(f"AllToAll   size={size_kb}KB  time={int(t)}us")
        except Exception as e:
            print(f"[rank {rank}] AllToAll {size_elems} failed: {e}")

        del in_buf, out_buf
        torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()

    if rank != 0:
        return None

    import pandas as pd
    ar_csv = pd.DataFrame(ar_rows, columns=header).to_csv(index=False).encode()
    a2a_csv = pd.DataFrame(a2a_rows, columns=header).to_csv(index=False).encode()
    return {"all_reduce.csv": ar_csv, "alltoall.csv": a2a_csv}


@app.local_entrypoint()
def main(out_dir: str = "comm_modal_out"):
    import pathlib

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Launching clustered run: {NUM_NODES} nodes x {GPUS_PER_NODE} {GPU}")
    results = profile_node.remote()
    if results is None:
        print("No results from rank 0 (cluster returned None).")
        return

    for name, blob in results.items():
        path = out / name
        path.write_bytes(blob)
        print(f"wrote {path}")
