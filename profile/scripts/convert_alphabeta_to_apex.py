"""Merge alpha-beta NCCL profile CSVs (from Modal sweep + NVL72 local sweep)
into apex's per-collective profile/comm/{gpu}/{op}.csv format.

Reads from both:
    profile/scripts/nccl_alphabeta_out/      — Modal sweep, worlds {2,4,8}
    profile/scripts/nccl_alphabeta_out_nvl72/ — local sweep, worlds {16,32,64,72}

Writes to:
    profile/comm/{gpu}/all_reduce.csv
    profile/comm/{gpu}/alltoall.csv
    profile/comm/{gpu}/all_gather.csv
    profile/comm/{gpu}/reduce_scatter.csv
    profile/comm/{gpu}/sendrecv.csv

Run from apex_plus repo root:
    uv run python profile/scripts/convert_alphabeta_to_apex.py
"""
import os
import pandas as pd

OP_TO_FILENAME = {
    "allreduce":     "all_reduce.csv",
    "alltoall":      "alltoall.csv",
    "allgather":     "all_gather.csv",
    "reducescatter": "reduce_scatter.csv",
    "sendrecv":      "sendrecv.csv",
}

SRC_DIRS = [
    "profile/scripts/nccl_alphabeta_out",
    "profile/scripts/nccl_alphabeta_out_nvl72",
]


def load_all_alphabeta() -> pd.DataFrame:
    """Concatenate every *_alphabeta.csv from each source dir."""
    dfs = []
    for src in SRC_DIRS:
        if not os.path.isdir(src):
            continue
        for fn in sorted(os.listdir(src)):
            if not fn.endswith("_alphabeta.csv"):
                continue
            dfs.append(pd.read_csv(os.path.join(src, fn)))
    if not dfs:
        raise SystemExit(
            f"No *_alphabeta.csv found in any of: {SRC_DIRS}"
        )
    return pd.concat(dfs, ignore_index=True)


def main():
    df = load_all_alphabeta()

    # Drop any duplicate (gpu, op, world, size) rows — later (NVL72) sweep
    # wins over earlier (Modal) for the same key, since rack-scale data is
    # the source of truth at large worlds.
    df = df.drop_duplicates(
        subset=["gpu", "op", "world_size", "size_bytes", "dtype"],
        keep="last",
    )

    written = []
    for gpu, gpu_df in df.groupby("gpu"):
        out_dir = f"profile/comm/{gpu}"
        os.makedirs(out_dir, exist_ok=True)

        for op, op_filename in OP_TO_FILENAME.items():
            sub = gpu_df[gpu_df["op"] == op].copy()
            if sub.empty:
                continue
            out = pd.DataFrame({
                "gpu":               sub["gpu"],
                "num_nodes":         1,
                "num_gpus_per_node": sub["world_size"],
                "dtype":             sub["dtype"],
                "size(kb)":          (sub["size_bytes"] // 1024).astype(int),
                "time(us)":          sub["time_us"].astype(int),
            })
            out = out.sort_values(by=["num_gpus_per_node", "dtype", "size(kb)"])
            out_path = os.path.join(out_dir, op_filename)
            out.to_csv(out_path, index=False)
            written.append((out_path, len(out)))

    print("Wrote:")
    for p, n in written:
        print(f"  {p}  ({n} rows)")


if __name__ == "__main__":
    main()
