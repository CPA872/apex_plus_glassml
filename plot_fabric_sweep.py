"""Bar charts for the fabric sweep.

Reads results.json produced by run_fabric_sweep.py and draws two figures:

1. fabric_comm_bars.png — communication time per fabric. NVLink and IB are
   solid bars (one comm-type breakdown each). Spectra bars (one per plane
   count) are stacked: lower = pure-comm time (delta=0 makespan), upper =
   reconfig overhead (delta=10 makespan minus delta=0 makespan).

2. fabric_total_bars.png — end-to-end iteration time per fabric, split into
   compute (MQA + SwiMoE + ...) vs communication. Useful for seeing how
   much fabric choice moves the wall-clock needle.

Usage:
  uv run python plot_fabric_sweep.py --input logs/fabric_sweep/results.json
  uv run python plot_fabric_sweep.py --input ... --out-dir plots/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMM_NAMES = ("AllReduce", "AllGather", "ReduceScatter", "AllToAll", "SendRecv")
COMM_COLORS = {
    "AllToAll":      "#2E86AB",
    "AllReduce":     "#A23B72",
    "AllGather":     "#F18F01",
    "ReduceScatter": "#C73E1D",
    "SendRecv":      "#5E6572",
}
RECONFIG_COLOR = "#7FB069"   # Reconfig overhead (top stack on spectra bars)
COMPUTE_COLOR = "#D3D3D3"    # Compute time (bottom of total-time chart)


def _comm_total(breakdown: Dict[str, float]) -> float:
    return sum(v for k, v in breakdown.items() if k in COMM_NAMES)


def _compute_total(breakdown: Dict[str, float]) -> float:
    return sum(v for k, v in breakdown.items() if k not in COMM_NAMES and k != "Idle")


def _split_spectra(results: List[dict]) -> Tuple[
    List[dict],  # non-spectra
    Dict[int, dict],  # planes -> default-delta result
    Dict[int, dict],  # planes -> delta=0 result
]:
    others: List[dict] = []
    default: Dict[int, dict] = {}
    zero: Dict[int, dict] = {}
    for r in results:
        m = re.match(r"^spectra s=(\d+)( d=0)?$", r["label"])
        if m is None:
            others.append(r)
            continue
        s = int(m.group(1))
        if m.group(2):
            zero[s] = r
        else:
            default[s] = r
    return others, default, zero


def plot_comm_bars(results: List[dict], out_path: Path,
                   reconfig_delay_us: float) -> None:
    others, default, zero = _split_spectra(results)
    plane_counts = sorted(default.keys())

    bar_labels: List[str] = []
    # Each column entry: list of (segment_label, height) tuples, plotted bottom-up.
    columns: List[List[Tuple[str, float]]] = []

    for r in others:
        bar_labels.append(r["label"])
        segs = []
        for c in COMM_NAMES:
            v = r["breakdown"].get(c, 0.0)
            if v > 0:
                segs.append((c, v))
        columns.append(segs)

    for s in plane_counts:
        d_default = default[s]
        d_zero = zero.get(s)
        full_a2a = d_default["breakdown"].get("AllToAll", 0.0)
        if d_zero is not None:
            comm_only = d_zero["breakdown"].get("AllToAll", 0.0)
            reconfig = max(full_a2a - comm_only, 0.0)
        else:
            comm_only, reconfig = full_a2a, 0.0
        bar_labels.append(f"spectra\ns={s}")
        # Bottom: pure comm. Top: reconfig overhead.
        segs = [("AllToAll", comm_only)]
        if reconfig > 0:
            segs.append(("Reconfig", reconfig))
        columns.append(segs)

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(bar_labels) + 3), 5.5))
    x = np.arange(len(bar_labels))
    width = 0.7

    seen_legend = set()
    for i, segs in enumerate(columns):
        bottom = 0.0
        for name, h in segs:
            color = RECONFIG_COLOR if name == "Reconfig" else COMM_COLORS.get(name, "#888")
            label = name if name not in seen_legend else None
            ax.bar(x[i], h, width, bottom=bottom, color=color,
                   edgecolor="black", linewidth=0.5, label=label)
            seen_legend.add(name)
            bottom += h
        # Total label on top.
        if bottom > 0:
            ax.text(x[i], bottom * 1.01, f"{bottom:.1f}s",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Communication time per iteration (s)")
    ax.set_title("Fabric comparison — DeepSeek-V3 EP=64\n"
                 f"Spectra bars: bottom = comm, top = reconfig (δ={reconfig_delay_us:g} µs)")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectra_zoom(results: List[dict], out_path: Path,
                      reconfig_delay_us: float) -> None:
    """Zoomed view: only spectra columns, so the reconfig stack is readable."""
    _, default, zero = _split_spectra(results)
    plane_counts = sorted(default.keys())
    if not plane_counts:
        return

    bar_labels: List[str] = []
    comm_only_h: List[float] = []
    reconfig_h: List[float] = []
    for s in plane_counts:
        d_default = default[s]
        d_zero = zero.get(s)
        full_a2a = d_default["breakdown"].get("AllToAll", 0.0)
        if d_zero is None:
            comm_only_h.append(full_a2a)
            reconfig_h.append(0.0)
        else:
            comm = d_zero["breakdown"].get("AllToAll", 0.0)
            comm_only_h.append(comm)
            reconfig_h.append(max(full_a2a - comm, 0.0))
        bar_labels.append(f"s={s}")

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(bar_labels) + 3), 5.0))
    x = np.arange(len(bar_labels))
    width = 0.6

    ax.bar(x, comm_only_h, width, color=COMM_COLORS["AllToAll"],
           edgecolor="black", linewidth=0.5, label="Pure comm (δ=0)")
    ax.bar(x, reconfig_h, width, bottom=comm_only_h, color=RECONFIG_COLOR,
           edgecolor="black", linewidth=0.5,
           label=f"Reconfig overhead (δ={reconfig_delay_us:g} µs)")

    for i, (c, r) in enumerate(zip(comm_only_h, reconfig_h)):
        total = c + r
        ax.text(x[i], total * 1.01, f"{total:.2f}s", ha="center", va="bottom", fontsize=9)
        if r > 0:
            ax.text(x[i], c + r / 2, f"{r:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_xlabel("Spectra plane count")
    ax.set_ylabel("AllToAll time per iteration (s)")
    ax.set_title("Spectra A2A — pure-comm vs reconfig overhead")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_total_bars(results: List[dict], out_path: Path) -> None:
    """End-to-end time, split into compute vs comm. δ=0 spectra runs hidden."""
    visible = [r for r in results if not r["label"].endswith("d=0")]
    bar_labels = [r["label"].replace(" (", "\n(") for r in visible]

    compute = np.array([_compute_total(r["breakdown"]) for r in visible])
    comm = np.array([_comm_total(r["breakdown"]) for r in visible])
    total = np.array([r["total_sec"] for r in visible])
    idle = total - compute - comm

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(bar_labels) + 3), 5.5))
    x = np.arange(len(bar_labels))
    width = 0.7

    ax.bar(x, compute, width, color=COMPUTE_COLOR, edgecolor="black",
           linewidth=0.5, label="Compute (MQA + MoE + ...)")
    ax.bar(x, comm, width, bottom=compute, color=COMM_COLORS["AllToAll"],
           edgecolor="black", linewidth=0.5, label="Communication")
    if (idle > 0.05).any():
        ax.bar(x, idle, width, bottom=compute + comm, color="#EEEEEE",
               edgecolor="black", linewidth=0.5, label="Idle")

    for i, t in enumerate(total):
        ax.text(x[i], t * 1.01, f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("End-to-end time per iteration (s)")
    ax.set_title("End-to-end iteration time — compute vs communication")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to results.json produced by run_fabric_sweep.py")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for PNGs (default: input's directory).")
    args = p.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text())
    results = data["results"]
    delta = data.get("config", {}).get("reconfig_delay_us", 10.0)

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    comm_path = out_dir / "fabric_comm_bars.png"
    zoom_path = out_dir / "fabric_spectra_zoom.png"
    total_path = out_dir / "fabric_total_bars.png"

    plot_comm_bars(results, comm_path, reconfig_delay_us=delta)
    plot_spectra_zoom(results, zoom_path, reconfig_delay_us=delta)
    plot_total_bars(results, total_path)

    print(f"Wrote {comm_path}")
    print(f"Wrote {zoom_path}")
    print(f"Wrote {total_path}")


if __name__ == "__main__":
    main()
