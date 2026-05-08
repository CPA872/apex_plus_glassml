"""Aggregate Modal gemm shard CSVs into the simulator's gemm_<freq>.csv format.

Reads every shard CSV from a directory (default: ./gemm_shards), groups rows
by graph_clk_freq (the SM clock — that's what the simulator's filenames key
on), drops the power/freq/energy columns, and writes one CSV per frequency:

    apex_plus/profile/comp/<gpu>/gemm_<graph_clk_freq>.csv

Schema written matches the existing comp/H100-SXM-80GB/gemm_*.csv files:
    gpu,dtype,m,k,n,time(us)

Since the Modal containers don't pin clocks (no nvidia-smi -ac in unprivileged
containers), there's typically just one frequency value per run, so you'll get
one output file. Multi-frequency sweeps from native gemm.py runs work too.
"""
import argparse
import pathlib
import sys

import pandas as pd


SIM_COLUMNS = ["gpu", "dtype", "m", "k", "n", "time(us)"]


def aggregate(shards_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    shard_files = sorted(shards_dir.glob("*.csv"))
    if not shard_files:
        sys.exit(f"No CSVs found in {shards_dir}")

    frames = [pd.read_csv(p) for p in shard_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(shard_files)} shards in {shards_dir}")

    missing = [c for c in SIM_COLUMNS + ["graph_clk_freq"] if c not in df.columns]
    if missing:
        sys.exit(f"Shard CSVs missing required columns: {missing}")

    df = df.sort_values(by=["gpu", "dtype", "m", "k", "n", "graph_clk_freq"])
    df = df.drop_duplicates(subset=["gpu", "dtype", "m", "k", "n", "graph_clk_freq"])

    output_dir.mkdir(parents=True, exist_ok=True)
    for freq, group in df.groupby("graph_clk_freq"):
        out_path = output_dir / f"gemm_{int(freq)}.csv"
        group[SIM_COLUMNS].to_csv(out_path, index=False)
        print(f"  freq={int(freq)} MHz -> {len(group)} rows -> {out_path}")


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    profile_root = here.parent  # apex_plus/profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-dir",
        type=pathlib.Path,
        default=pathlib.Path("gemm_shards"),
        help="Directory containing per-shard CSVs from gemm_modal.py.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="B200-SXM-192GB",
        help="GPU label; selects the comp/<gpu>/ subdir to write into.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Override the output directory. Default: "
            "<repo>/apex_plus/profile/comp/<gpu>/"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (profile_root / "comp" / args.gpu)
    aggregate(args.shards_dir.resolve(), output_dir.resolve())


if __name__ == "__main__":
    main()
