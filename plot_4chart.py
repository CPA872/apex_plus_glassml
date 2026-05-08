"""2×2 grid of fabric A2A bar charts: (uniform vs real) × (8k vs 128k).

Loads four results.json files produced by run_dm_replay.py and renders
them as a single 2×2 figure where each subplot has one bar per fabric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLOR_NVLINK = "#4C72B0"
COLOR_IB = "#C44E52"
COLOR_SPECTRA = "#55A868"


def pick_unit(values_s: List[float]) -> Tuple[str, float]:
    m = max(values_s)
    if m >= 1.0:
        return "s", 1.0
    if m >= 1e-3:
        return "ms", 1e3
    return "µs", 1e6


def bar_color(label: str) -> str:
    if "nvlink" in label:
        return COLOR_NVLINK
    if label.startswith("ib"):
        return COLOR_IB
    return COLOR_SPECTRA


def load_results(path: Path) -> Tuple[List[str], List[float], dict]:
    data = json.loads(path.read_text())
    cfg = data["config"]
    results = data["results"]
    labels = [r["label"] for r in results]
    times_s = [r["total_sec"] for r in results]
    return labels, times_s, cfg


def render_subplot(ax, labels: List[str], times_s: List[float],
                   unit: str, mult: float, title: str, subtitle: str) -> None:
    x = np.arange(len(labels))
    heights = [t * mult for t in times_s]
    colors = [bar_color(lab) for lab in labels]

    ax.bar(x, heights, 0.7, color=colors, edgecolor="black", linewidth=0.5)
    for xi, h in zip(x, heights):
        ax.text(xi, h * 1.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    pretty = []
    for lab in labels:
        if lab.startswith("spectra "):
            pretty.append(lab.replace("spectra ", "").strip())
        elif "nvlink" in lab:
            pretty.append("NVLink")
        else:
            pretty.append("IB")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=9)
    ax.set_ylabel(f"AllToAll time ({unit})", fontsize=10)
    full_title = title + ("\n" + subtitle if subtitle else "")
    ax.set_title(full_title, fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(heights) * 1.15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform-8k", required=True)
    p.add_argument("--uniform-128k", required=True)
    p.add_argument("--real-8k", required=True)
    p.add_argument("--real-128k", required=True)
    p.add_argument("--out", default="logs/4chart/fabric_4chart.png")
    args = p.parse_args()

    cells = [
        ("Uniform A2A — seq=8K",   args.uniform_8k,   "uniform"),
        ("Uniform A2A — seq=128K", args.uniform_128k, "uniform"),
        ("Real A2A (Qwen3-235B) — seq=8K",   args.real_8k,   "real"),
        ("Real A2A (Qwen3-235B) — seq=128K", args.real_128k, "real"),
    ]

    loaded = []
    all_times: List[float] = []
    for title, path, kind in cells:
        labels, times_s, cfg = load_results(Path(path))
        loaded.append((title, labels, times_s, cfg, kind))
        all_times.extend(times_s)

    # Use a consistent unit per-row (seqlen) so columns are comparable.
    # Top row (8k) and bottom row (128k) get their own unit.
    unit_8k, mult_8k = pick_unit([t for tup in loaded[::2] for t in tup[2]])
    unit_128k, mult_128k = pick_unit([t for tup in loaded[1::2] for t in tup[2]])
    units = [(unit_8k, mult_8k), (unit_128k, mult_128k),
             (unit_8k, mult_8k), (unit_128k, mult_128k)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    flat_axes = axes.flatten()

    for ax, (title, labels, times_s, cfg, kind), (unit, mult) in zip(flat_axes, loaded, units):
        skew = cfg.get("skew", {})
        if kind == "uniform":
            sub = (f"total {skew.get('total_MB', 0):.0f} MB "
                   f"(row=col={skew.get('row_max_over_mean', 0):.2f}× max/mean — perfectly uniform)")
        else:
            sub = (f"total {skew.get('total_MB', 0):.0f} MB | "
                   f"row={skew.get('row_max_over_mean', 0):.2f}× / "
                   f"col={skew.get('col_max_over_mean', 0):.2f}× max/mean")
        render_subplot(ax, labels, times_s, unit, mult, title, sub)

    fig.suptitle("Fabric A2A comparison — N=64 ranks, EP=64, "
                 "NVLink 900 GB/s · IB 8×50 GB/s rails · Spectra 8×1024 Gbps WGs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
