"""Bar chart for run_dm_replay.py — single-layer A2A across fabrics.

Reads the JSON output of run_dm_replay.py and draws one chart with
auto-scaled units (s/ms/µs) so single-layer numbers stay readable.

Usage:
  uv run python plot_dm_replay.py --input logs/dm_replay_qwen235b/results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pick_unit(seconds_max: float) -> Tuple[str, float]:
    """Return (unit_label, multiplier_to_apply_to_seconds_value)."""
    if seconds_max >= 1.0:
        return "s", 1.0
    if seconds_max >= 1e-3:
        return "ms", 1e3
    return "µs", 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: alongside input).")
    p.add_argument("--title", default=None,
                   help="Chart title; defaults to a description from the JSON config.")
    args = p.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text())
    cfg = data.get("config", {})
    results = data["results"]

    labels = [r["label"] for r in results]
    times_s = [r["total_sec"] for r in results]

    unit, mult = pick_unit(max(times_s))
    heights = [t * mult for t in times_s]

    # Distinguish nvlink/ib (one color) from spectra (a gradient by plane count).
    colors: List[str] = []
    for label in labels:
        if "nvlink" in label:
            colors.append("#4C72B0")
        elif "ib" in label:
            colors.append("#C44E52")
        else:
            colors.append("#55A868")

    # Build title from config.
    if args.title:
        title = args.title
    else:
        layer_file = cfg.get("layer_file", "")
        skew = cfg.get("skew", {})
        N = cfg.get("matrix_size", "?")
        bits = []
        m = re.search(r"(ep\d+)", layer_file)
        if m:
            bits.append(m.group(1).upper())
        m = re.search(r"(\d+k)", layer_file)
        if m:
            bits.append(f"seq={m.group(1)}")
        bits.append(f"N={N}")
        title_main = "A2A on profiled demand matrix — " + ", ".join(bits)
        title_sub = (f"layer={layer_file} | total {skew.get('total_MB', 0):.0f} MB | "
                     f"row={skew.get('row_max_over_mean', 0):.2f}× / "
                     f"col={skew.get('col_max_over_mean', 0):.2f}× / "
                     f"pair={skew.get('pair_max_over_mean', 0):.2f}× max/mean")
        title = title_main + "\n" + title_sub

    fig, ax = plt.subplots(figsize=(max(7, 0.85 * len(labels) + 2), 5.0))
    x = np.arange(len(labels))
    width = 0.65

    bars = ax.bar(x, heights, width, color=colors, edgecolor="black", linewidth=0.5)

    for xi, h in zip(x, heights):
        ax.text(xi, h * 1.01, f"{h:.2f} {unit}",
                ha="center", va="bottom", fontsize=9)

    # X-tick labels: split spectra ones onto two lines.
    pretty = []
    for lab in labels:
        if lab.startswith("spectra "):
            pretty.append(lab.replace("spectra ", "spectra\n"))
        else:
            pretty.append(lab)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylabel(f"AllToAll time ({unit})")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Headroom for value labels.
    ax.set_ylim(0, max(heights) * 1.12)

    fig.tight_layout()
    out = Path(args.out) if args.out else in_path.parent / "dm_replay_bars.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
