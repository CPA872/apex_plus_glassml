"""Fit α-β model to NCCL profile data and plot latency/bandwidth regimes.

Reads CSVs produced by nccl_alphabeta_modal.py and:
  1. Fits T(S) = α + S/β per (gpu, op, world).
  2. Prints (α µs, β GB/s) and the knee message size = α·β.
  3. Plots a log-log scatter + fit line per (op, world), one PNG per GPU.

α is estimated as the minimum measured time at the smallest size (latency
plateau). β is estimated from the slope of the two largest sizes (bandwidth
plateau). This is robust against the curve fitter getting confused by the
strong asymmetry in T(S) across regimes.

Usage:
    cd apex_plus/profile/scripts
    uv run python plot_alphabeta.py --in-dir nccl_alphabeta_out
"""
import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def fit_alphabeta(sizes_bytes: np.ndarray, times_us: np.ndarray) -> Tuple[float, float]:
    """Return (α µs, β GB/s) from sparse log-spaced (S, T) samples.

    α  ≈ T at smallest S (latency plateau).
    β  ≈ slope of (T vs S) at largest two S (bandwidth plateau), inverted.

    Bandwidth model: T_us = α_us + S_bytes / β_(bytes/µs)
        ⇒ β_(bytes/µs) = (S_max - S_2nd) / (T_max - T_2nd)
        ⇒ β_GBps      = β_(bytes/µs) * 1e-3
    """
    order = np.argsort(sizes_bytes)
    s = sizes_bytes[order]
    t = times_us[order]

    # α = the minimum of the latency-bound region. Take the smaller of T at
    # the smallest 1-2 sizes; cap at 0 to avoid negative α from noise.
    alpha = max(float(np.min(t[: max(1, len(t) // 4)])), 0.0)

    # β slope from the two largest sizes (bandwidth plateau).
    if len(s) >= 2 and t[-1] > t[-2] + 1e-9:
        bytes_per_us = (s[-1] - s[-2]) / (t[-1] - t[-2])
    else:
        bytes_per_us = s[-1] / max(t[-1], 1e-9)

    beta_gbps = bytes_per_us * 1e-3  # bytes/µs → GB/s
    return alpha, beta_gbps


def plot_merged(
    df: pd.DataFrame, out_path: str
) -> Dict[Tuple[str, str, int], Tuple[float, float]]:
    """One PNG with all GPUs overlaid.

    Layout: 1 row × N columns, one panel per collective (op). Each panel
    shows curves for every (gpu, world_size) combination — solid for one GPU,
    dashed for the other (or distinguished by marker/color).
    """
    ops = sorted(df["op"].unique())
    gpus = sorted(df["gpu"].unique())
    worlds = sorted(df["world_size"].unique())

    fig, axes = plt.subplots(1, len(ops), figsize=(7 * len(ops), 6), squeeze=False)

    fits: Dict[Tuple[str, str, int], Tuple[float, float]] = {}

    # Color per world, linestyle/marker per GPU — keeps both axes legible.
    world_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(worlds)))
    gpu_styles = {
        gpus[0]: {"linestyle": "-", "marker": "o", "fillstyle": "full"},
        gpus[1] if len(gpus) > 1 else "_dummy": {
            "linestyle": "--", "marker": "s", "fillstyle": "none",
        },
    }

    for col_idx, op in enumerate(ops):
        ax = axes[0][col_idx]
        for w_idx, w in enumerate(worlds):
            color = world_colors[w_idx]
            for gpu in gpus:
                sub = df[
                    (df["op"] == op) & (df["world_size"] == w) & (df["gpu"] == gpu)
                ].sort_values("size_bytes")
                if sub.empty:
                    continue
                s = sub["size_bytes"].to_numpy(dtype=float)
                t = sub["time_us"].to_numpy(dtype=float)

                alpha, beta_gbps = fit_alphabeta(s, t)
                fits[(gpu, op, int(w))] = (alpha, beta_gbps)

                style = gpu_styles[gpu]
                gpu_short = gpu.split("-")[0]
                ax.loglog(
                    s, t,
                    marker=style["marker"],
                    fillstyle=style["fillstyle"],
                    linestyle=style["linestyle"],
                    color=color,
                    linewidth=1.2,
                    markersize=5,
                    label=f"{gpu_short} world={w}",
                )

                # α-β model curve.
                s_dense = np.geomspace(s[0], s[-1], 200)
                t_model = alpha + s_dense / (beta_gbps * 1e3)
                ax.loglog(
                    s_dense, t_model,
                    linestyle=":",
                    color=color,
                    alpha=0.35,
                    linewidth=0.8,
                )

        ax.set_title(f"{op}  —  log-log T vs S")
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("time (µs)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.suptitle(
        "NCCL collectives: latency (flat) vs bandwidth (slope=1) regimes",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fits


def print_fits(fits: Dict[Tuple[str, str, int], Tuple[float, float]]):
    print(
        f"{'gpu':>22s}  {'op':>10s}  {'world':>5s}  "
        f"{'α (µs)':>10s}  {'β (GB/s)':>12s}  {'knee (MB)':>12s}"
    )
    for (gpu, op, w) in sorted(fits.keys()):
        alpha, beta = fits[(gpu, op, w)]
        # knee bytes = α [µs] · β [GB/s] · 1e3 (B/µs per GB/s)
        knee_mb = alpha * beta * 1e3 / (1 << 20)
        print(
            f"{gpu:>22s}  {op:>10s}  {w:>5d}  "
            f"{alpha:>10.2f}  {beta:>12.1f}  {knee_mb:>12.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="nccl_alphabeta_out")
    parser.add_argument("--out-dir", default="nccl_alphabeta_plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = sorted(f for f in os.listdir(args.in_dir) if f.endswith("_alphabeta.csv"))
    if not csvs:
        raise SystemExit(f"No *_alphabeta.csv files found in {args.in_dir}")

    # Merge all GPUs' CSVs into one DataFrame for the combined plot.
    dfs = [pd.read_csv(os.path.join(args.in_dir, f)) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)

    out_png = os.path.join(args.out_dir, "alphabeta_merged.png")
    fits = plot_merged(df, out_png)
    print_fits(fits)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
