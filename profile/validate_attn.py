#!/usr/bin/env python3
"""Plot measured attention profile sanity checks from a profiler CSV.

Supports the legacy mha_*.csv schema and the newer FlashAttention attn.csv
schema. The roofline is inferred only from the CSV: measured peak TFLOP/s and
measured effective bandwidth are used as the two ceilings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DTYPE_BYTES = {
    "half": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float": 4,
    "fp32": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw attention roofline-style sanity plots from a profiled CSV."
    )
    parser.add_argument("csv", type=Path, help="Path to an attention CSV, e.g. comp/H100-SXM-80GB/mha_0.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output image path. Defaults to <csv>.sanity.png")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window after saving")
    parser.add_argument("--max-points", type=int, default=200_000, help="Maximum scatter points to draw")
    parser.add_argument(
        "--assume-causal",
        action="store_true",
        help="Treat legacy CSVs with no causal column as causal attention",
    )
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


def full_pairs(seq_len: pd.Series, causal: pd.Series) -> pd.Series:
    seq = seq_len.astype(float)
    return np.where(causal.astype(bool), seq * (seq + 1.0) / 2.0, seq * seq)


def causal_window_pairs(seq_len: pd.Series, window_left: pd.Series) -> pd.Series:
    seq = seq_len.astype(float)
    window = (window_left.astype(float) + 1.0).clip(lower=1.0)
    return np.where(
        seq <= window,
        seq * (seq + 1.0) / 2.0,
        window * seq - window * (window - 1.0) / 2.0,
    )


def effective_attention_pairs(df: pd.DataFrame) -> pd.Series:
    seq = df["seq_len"].astype(float)
    causal = df["causal"].astype(bool)
    left = df["window_left"].astype(float)
    right = df["window_right"].astype(float)

    pairs = pd.Series(full_pairs(seq, causal), index=df.index, dtype=float)

    causal_left_only = causal & (left >= 0) & (right <= 0)
    if causal_left_only.any():
        pairs.loc[causal_left_only] = causal_window_pairs(
            seq.loc[causal_left_only], left.loc[causal_left_only]
        )

    finite_noncausal_window = (~causal) & (left >= 0) & (right >= 0)
    if finite_noncausal_window.any():
        window = (left.loc[finite_noncausal_window] + right.loc[finite_noncausal_window] + 1.0).clip(lower=1.0)
        pairs.loc[finite_noncausal_window] = np.minimum(
            seq.loc[finite_noncausal_window] * seq.loc[finite_noncausal_window],
            seq.loc[finite_noncausal_window] * window,
        )

    return pairs


def enrich(df: pd.DataFrame, csv_path: Path, assume_causal: bool) -> pd.DataFrame:
    require_columns(
        df,
        ["num_heads", "head_size", "batch_size", "seq_len", "time(us)", "dtype"],
        csv_path,
    )
    df = df.copy()
    if "num_heads_kv" not in df.columns:
        df["num_heads_kv"] = df["num_heads"]
    if "causal" not in df.columns:
        df["causal"] = int(assume_causal)
    if "window_left" not in df.columns:
        df["window_left"] = -1
    if "window_right" not in df.columns:
        df["window_right"] = -1
    if "variant" not in df.columns:
        df["variant"] = "legacy"

    numeric_cols = [
        "num_heads",
        "num_heads_kv",
        "head_size",
        "batch_size",
        "seq_len",
        "time(us)",
        "causal",
        "window_left",
        "window_right",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    df = df[df["time(us)"] > 0]
    if df.empty:
        raise SystemExit("No valid attention rows with positive time(us).")

    bytes_per_elem = dtype_nbytes(df["dtype"])
    h = df["num_heads"].astype(float)
    h_kv = df["num_heads_kv"].astype(float)
    d = df["head_size"].astype(float)
    b = df["batch_size"].astype(float)
    s = df["seq_len"].astype(float)
    time_s = df["time(us)"].astype(float) * 1e-6

    pairs = effective_attention_pairs(df)
    df["attention_pairs_per_head"] = pairs
    df["flops"] = 4.0 * b * h * pairs * d
    df["bytes"] = (b * s * d * (2.0 * h + 2.0 * h_kv)) * bytes_per_elem
    df["ai_flop_per_byte"] = df["flops"] / df["bytes"].replace(0, np.nan)
    df["tflops"] = df["flops"] / time_s / 1e12
    df["effective_tbps"] = df["bytes"] / time_s / 1e12
    df["tokens"] = b * s
    df["q_elems"] = b * h * s * d
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ai_flop_per_byte", "tflops", "effective_tbps", "q_elems"]
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
    title_bits.append(",".join(map(str, sorted(df["variant"].dropna().unique()))))
    title = "Attention profile sanity: " + " / ".join(bit for bit in title_bits if bit)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    sc = ax.scatter(
        points["ai_flop_per_byte"],
        points["tflops"],
        c=points["q_elems"],
        s=8,
        alpha=0.45,
        cmap="viridis",
        norm=LogNorm(vmin=max(points["q_elems"].min(), 1), vmax=points["q_elems"].max()),
        linewidths=0,
    )
    add_roofline(ax, df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Measured throughput (TFLOP/s)")
    ax.set_title("Observed roofline")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="B*H*S*D query elements")

    ax = axes[0, 1]
    sc = ax.scatter(
        points["seq_len"],
        points["tflops"],
        c=points["tokens"],
        s=8,
        alpha=0.45,
        cmap="plasma",
        norm=LogNorm(vmin=max(points["tokens"].min(), 1), vmax=points["tokens"].max()),
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Measured throughput (TFLOP/s)")
    ax.set_title("Machine FLOP/s vs sequence length")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="batch_size*seq_len")

    ax = axes[1, 0]
    sc = ax.scatter(
        points["seq_len"],
        points["ai_flop_per_byte"],
        c=points["head_size"],
        s=8,
        alpha=0.45,
        cmap="magma",
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
    ax.set_title("Arithmetic intensity vs attention dimension")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="head_size")

    ax = axes[1, 1]
    heat = (
        df.groupby(["head_size", "seq_len"], as_index=False)["tflops"]
        .max()
        .pivot(index="head_size", columns="seq_len", values="tflops")
        .sort_index()
        .sort_index(axis=1)
    )
    im = ax.imshow(heat.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    xlabels = sparse_labels(list(heat.columns), max_labels=28)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([format_dim(v) for v in heat.index], fontsize=8)
    ax.set_xlabel("seq_len")
    ax.set_ylabel("head_size")
    ax.set_title("Best measured TFLOP/s over batch/heads")
    fig.colorbar(im, ax=ax, label="TFLOP/s")

    fig.text(0.01, 0.01, f"source: {csv_path} | rows: {len(df):,}", fontsize=8, color="0.35")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def sparse_labels(values: list[float], max_labels: int) -> list[str]:
    if len(values) <= max_labels:
        return [format_dim(v) for v in values]
    stride = int(np.ceil(len(values) / max_labels))
    return [format_dim(v) if i % stride == 0 else "" for i, v in enumerate(values)]


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
        f"B={int(best['batch_size'])} H={int(best['num_heads'])} "
        f"Hkv={int(best['num_heads_kv'])} S={int(best['seq_len'])} "
        f"D={int(best['head_size'])} dtype={best['dtype']} "
        f"causal={int(best['causal'])} time={best['time(us)']:.3g} us"
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = enrich(df, args.csv, args.assume_causal)
    out_path = args.out or default_out(args.csv)
    plot(df, args.csv, out_path, args.show, args.max_points)
    print_summary(df, out_path)


if __name__ == "__main__":
    main()
