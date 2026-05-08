#!/usr/bin/env python3
"""Plot measured GEMM profile sanity checks from a profiler CSV.

The roofline is inferred only from the CSV: the horizontal ceiling is the
maximum measured TFLOP/s and the sloped ceiling is the maximum measured
effective bandwidth from the same rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DTYPE_BYTES = {
    "fp8": 1,
    "fp8_e4m3": 1,
    "float8": 1,
    "half": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float": 4,
    "fp32": 4,
    "double": 8,
    "fp64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw GEMM roofline-style sanity plots from a profiled CSV."
    )
    parser.add_argument("csv", type=Path, help="Path to a GEMM CSV, e.g. comp/H100-SXM-80GB/gemm_0.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output image path. Defaults to <csv>.sanity.png")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window after saving")
    parser.add_argument("--max-points", type=int, default=200_000, help="Maximum scatter points to draw")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], csv_path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {', '.join(missing)}")


def dtype_nbytes(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.lower()
    out = normalized.map(DTYPE_BYTES)
    if out.isna().any():
        unknown = sorted(normalized[out.isna()].unique())
        raise SystemExit(f"Unknown dtype byte width for: {', '.join(unknown)}")
    return out.astype(float)


def enrich(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    require_columns(df, ["m", "k", "n", "time(us)", "dtype"], csv_path)
    df = df.copy()

    numeric_cols = ["m", "k", "n", "time(us)"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    df = df[df["time(us)"] > 0]
    if df.empty:
        raise SystemExit("No valid GEMM rows with positive time(us).")

    bytes_per_elem = dtype_nbytes(df["dtype"])
    m = df["m"].astype(float)
    k = df["k"].astype(float)
    n = df["n"].astype(float)
    time_s = df["time(us)"].astype(float) * 1e-6

    df["flops"] = 2.0 * m * k * n
    df["bytes"] = (m * k + k * n + m * n) * bytes_per_elem
    df["ai_flop_per_byte"] = df["flops"] / df["bytes"].replace(0, np.nan)
    df["tflops"] = df["flops"] / time_s / 1e12
    df["effective_tbps"] = df["bytes"] / time_s / 1e12
    df["output_elems"] = m * n
    df["mnk"] = m * n * k
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ai_flop_per_byte", "tflops", "effective_tbps", "mnk"]
    )


def sample_points(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=0)


def default_out(csv_path: Path) -> Path:
    return csv_path.with_suffix("").with_name(csv_path.with_suffix("").name + ".sanity.png")


def add_roofline(ax, df: pd.DataFrame) -> None:
    peak_tflops = df["tflops"].max()
    peak_tbps = df["effective_tbps"].max()
    xmin = max(df["ai_flop_per_byte"].min() * 0.8, 1e-6)
    xmax = df["ai_flop_per_byte"].max() * 1.25
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 256)
    ys = np.minimum(peak_tflops, peak_tbps * xs)
    ax.plot(xs, ys, color="black", lw=1.8, label="Observed roofline")
    ax.axhline(peak_tflops, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.text(
        0.02,
        0.98,
        f"peak {peak_tflops:.2f} TFLOP/s\nbandwidth {peak_tbps:.2f} TB/s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "alpha": 0.9},
    )


def plot(df: pd.DataFrame, csv_path: Path, out_path: Path, show: bool, max_points: int) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    points = sample_points(df, max_points)
    title_bits = []
    if "gpu" in df.columns:
        title_bits.append(",".join(map(str, sorted(df["gpu"].dropna().unique())[:3])))
    title_bits.append(",".join(map(str, sorted(df["dtype"].dropna().unique()))))
    title = "GEMM profile sanity: " + " / ".join(bit for bit in title_bits if bit)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    sc = ax.scatter(
        points["ai_flop_per_byte"],
        points["tflops"],
        c=points["mnk"],
        s=8,
        alpha=0.45,
        cmap="viridis",
        norm=LogNorm(vmin=max(points["mnk"].min(), 1), vmax=points["mnk"].max()),
        linewidths=0,
    )
    add_roofline(ax, df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Measured throughput (TFLOP/s)")
    ax.set_title("Observed roofline")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="m*n*k")

    ax = axes[0, 1]
    sc = ax.scatter(
        points["mnk"],
        points["tflops"],
        c=points["ai_flop_per_byte"],
        s=8,
        alpha=0.45,
        cmap="plasma",
        norm=LogNorm(vmin=max(points["ai_flop_per_byte"].min(), 1e-6), vmax=points["ai_flop_per_byte"].max()),
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem volume m*n*k")
    ax.set_ylabel("Measured throughput (TFLOP/s)")
    ax.set_title("Machine FLOP/s vs GEMM size")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="FLOP/byte")

    ax = axes[1, 0]
    sc = ax.scatter(
        points["n"],
        points["ai_flop_per_byte"],
        c=points["tflops"],
        s=8,
        alpha=0.45,
        cmap="magma",
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax.set_title("Arithmetic intensity vs token/output dimension n")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="TFLOP/s")

    ax = axes[1, 1]
    heat = (
        df.groupby(["m", "k"], as_index=False)["tflops"]
        .max()
        .pivot(index="m", columns="k", values="tflops")
        .sort_index()
        .sort_index(axis=1)
    )
    im = ax.imshow(heat.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([format_dim(v) for v in heat.columns], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([format_dim(v) for v in heat.index], fontsize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("m")
    ax.set_title("Best measured TFLOP/s over n")
    fig.colorbar(im, ax=ax, label="TFLOP/s")

    fig.text(0.01, 0.01, f"source: {csv_path} | rows: {len(df):,}", fontsize=8, color="0.35")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def format_dim(value: float) -> str:
    value = float(value)
    if value >= 1024 and value.is_integer() and int(value) % 1024 == 0:
        return f"{int(value // 1024)}K"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def print_summary(df: pd.DataFrame, out_path: Path) -> None:
    best = df.loc[df["tflops"].idxmax()]
    print(f"Saved {out_path}")
    print(f"Rows plotted: {len(df):,}")
    print(f"Observed peak: {df['tflops'].max():.3f} TFLOP/s")
    print(f"Observed effective bandwidth ceiling: {df['effective_tbps'].max():.3f} TB/s")
    print(
        "Best shape: "
        f"m={int(best['m'])} k={int(best['k'])} n={int(best['n'])} "
        f"dtype={best['dtype']} time={best['time(us)']:.3g} us"
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = enrich(df, args.csv)
    out_path = args.out or default_out(args.csv)
    plot(df, args.csv, out_path, args.show, args.max_points)
    print_summary(df, out_path)


if __name__ == "__main__":
    main()
