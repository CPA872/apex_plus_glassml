"""Replay profiled per-layer demand matrices through each fabric model.

Reads N×N demand matrices (in bytes) from a directory and runs each one
through three fabric cost models — analytical NVLink (full bisection),
analytical IB (single-rail-per-pair), and the SPECTRA Julia solver — to
get a per-fabric all-to-all time per layer. Sums across layers and writes
a JSON suitable for the existing plot_fabric_sweep.py-style chart.

Source matrices (from glassml/profiling/qwen3-235b):
  output/ep64/8k/per_layer/traffic_matrix_ep64_8k_layer<NN>.txt

Each .txt is a JSON-ish list-of-lists of bytes-per-pair. One file per layer.

Cost models — per fabric, given a single layer's N×N demand matrix D:

  NVLink (NVL64-analytical, full bisection):
    Each GPU has per-GPU bandwidth `nvlink_bw_GBs`. All pairs can transmit
    simultaneously, so the bottleneck is the source rank with the largest
    out-degree-bytes:
        t = max_i (sum_j D[i,j]) / nvlink_bw_GBs

  IB (analytical, single-rail-per-pair):
    APEX's IB A2A model says ALL of a GPU's inter-node A2A traffic uses
    one NIC (no multi-rail splitting like ring collectives). With per-rail
    bandwidth `ib_rail_bw_GBs`:
        t = max_i (sum_j D[i,j]) / ib_rail_bw_GBs  +  ib_latency_us

  SPECTRA (Julia solver):
    D_us = D / per_port_bw_bytes_per_us
    perms, durations, makespan = spectra(D_us, s=num_planes, delta=...)
    t = makespan

Usage:
  uv run python run_dm_replay.py \\
      --matrices-dir /mnt/alpha/yuepan/glassml/profiling/qwen3-235b/output/ep64/8k/per_layer \\
      --planes 1,2,4,8,16 \\
      --logs logs/dm_replay_qwen235b_ep64_8k
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Defaults aligned with apex_plus/config.yaml + comm_profile.py.
DEFAULT_NVLINK_BW_GBs = 900.0      # H100 SXM per-GPU NVLink bandwidth
DEFAULT_IB_RAIL_BW_GBs = 50.0      # 400 Gbps = 50 GB/s per rail (ConnectX-7)
DEFAULT_IB_NUM_RAILS = 8           # DGX H100: 8 ConnectX-7 NICs per node, 1 per GPU
DEFAULT_IB_LATENCY_US = 3.0        # base IB RDMA latency
DEFAULT_WGS_PER_GPU = 8
DEFAULT_WG_SPEED_GBPS = 1024.0
DEFAULT_RECONFIG_DELAY_US = 10.0


@dataclass
class FabricResult:
    label: str
    per_layer_us: List[float]
    total_us: float


def load_matrix(path: Path) -> np.ndarray:
    """Load a demand matrix from a Python-literal text file (list-of-lists)."""
    text = path.read_text()
    # The profiled files use Python-literal syntax with trailing commas, so
    # use ast.literal_eval rather than json.loads.
    D = np.asarray(ast.literal_eval(text), dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{path} is not a square matrix; got shape {D.shape}")
    return D


def matrix_skew(D: np.ndarray) -> Dict[str, float]:
    """Quantify how non-uniform a demand matrix is.

    Returns row-sum, column-sum, and pair-level skew. For MoE A2A row sums are
    typically near-uniform (top-k routing balances source dispatch), while
    column sums are skewed by hot experts.
    """
    row_sums = D.sum(axis=1)
    col_sums = D.sum(axis=0)
    nonzero = D[D > 0]
    return {
        "row_max_over_mean": float(np.max(row_sums) / max(np.mean(row_sums), 1e-9)),
        "col_max_over_mean": float(np.max(col_sums) / max(np.mean(col_sums), 1e-9)),
        "pair_max_over_mean": float(np.max(D) / max(np.mean(nonzero), 1e-9))
                              if nonzero.size else 0.0,
        "cv": float(np.std(nonzero) / max(np.mean(nonzero), 1e-9))
              if nonzero.size else 0.0,
        "total_MB": float(D.sum() / 1e6),
    }


def discover_matrices(matrices_dir: Path) -> List[Path]:
    """Return the sorted list of per-layer demand-matrix .txt files."""
    paths = sorted(matrices_dir.glob("*.txt"))
    # Skip the aggregate "all_layers" file if it lives here.
    paths = [p for p in paths if "all_layers" not in p.name]
    if not paths:
        raise FileNotFoundError(f"No .txt files found under {matrices_dir}")
    return paths


def _row_col_max(D: np.ndarray) -> float:
    """Max of (max row sum, max col sum) — joint egress/ingress bottleneck."""
    return float(max(np.max(D.sum(axis=1)), np.max(D.sum(axis=0))))


def nvlink_time_us(D: np.ndarray, nvlink_bw_GBs: float) -> float:
    """Full-bisection A2A: bottleneck = max(busiest egress, busiest ingress)."""
    if D.size == 0 or float(D.sum()) == 0.0:
        return 0.0
    bw_bytes_per_us = nvlink_bw_GBs * 1e9 / 1e6
    return _row_col_max(D) / bw_bytes_per_us


def ib_time_us(D: np.ndarray, rail_bw_GBs: float, num_rails: int,
               latency_us: float) -> float:
    """A2A on IB with all `num_rails` rails per node usable per GPU.

    Models multi-rail bonded A2A: per-GPU bandwidth = num_rails × rail_bw,
    bottleneck = max(busiest egress, busiest ingress).
    """
    if D.size == 0 or float(D.sum()) == 0.0:
        return 0.0
    total_bw_GBs = rail_bw_GBs * num_rails
    bw_bytes_per_us = total_bw_GBs * 1e9 / 1e6
    return _row_col_max(D) / bw_bytes_per_us + latency_us


def _spectra_per_port_bw_bytes_per_us(num_planes: int, wgs_per_gpu: int,
                                      wg_speed_gbps: float) -> float:
    """(wgs_per_gpu / num_planes) × wg_speed_gbps / 8 GB/s, then × 1e3 for bytes/µs."""
    bonding = wgs_per_gpu / num_planes
    GBs = bonding * wg_speed_gbps / 8.0
    return GBs * 1e9 / 1e6


_SOLVER = None


def _get_solver():
    global _SOLVER
    if _SOLVER is not None:
        return _SOLVER
    here = Path(__file__).resolve().parent
    # mesh/spectra is nested under apex_plus/, matching apex_plus/simulator/spectra_sim.py.
    spectra_dir = here / "mesh" / "spectra"
    if not spectra_dir.exists():
        # Fall back to glassml/mesh/spectra if the layout changes.
        spectra_dir = here.parent / "mesh" / "spectra"
    sys.path.insert(0, str(spectra_dir))
    import spectra_solver  # type: ignore
    _SOLVER = spectra_solver
    return _SOLVER


def spectra_time_us(D: np.ndarray, num_planes: int, wgs_per_gpu: int,
                    wg_speed_gbps: float, delta_us: float) -> float:
    if D.size == 0 or float(D.sum()) == 0.0:
        return 0.0
    bw = _spectra_per_port_bw_bytes_per_us(num_planes, wgs_per_gpu, wg_speed_gbps)
    D_us = D / bw
    solver = _get_solver()
    _perms, _durations, makespan = solver.spectra(D_us, s=num_planes, delta=delta_us)
    return float(makespan)


def replay_fabric(label: str, matrices: List[np.ndarray], fn) -> FabricResult:
    t0 = time.time()
    per_layer = [fn(D) for D in matrices]
    total = sum(per_layer)
    elapsed = time.time() - t0
    print(f"  [{label:>15}]  total={total/1000:>10.3f} ms   "
          f"({len(matrices)} layers, {elapsed:.1f}s)")
    return FabricResult(label=label, per_layer_us=per_layer, total_us=total)


def make_uniform_dm(N: int, seqlen: int, hidden_size: int, topk: int,
                    dtype_bytes: int = 2) -> np.ndarray:
    """Construct a perfectly-uniform A2A demand matrix (in bytes).

    For MoE A2A with `seqlen` tokens per source rank, top-k routing, and
    experts uniformly distributed across N ranks, the dispatch demand per
    pair (i, j != i) is:
        per_pair_bytes = seqlen × topk / N × hidden_size × dtype_bytes
    Off-diagonal entries are equal; diagonal is zero.
    """
    per_pair_tokens = seqlen * topk / N
    per_pair_bytes = per_pair_tokens * hidden_size * dtype_bytes
    D = np.full((N, N), per_pair_bytes, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    return D


def pick_most_skewed(paths: List[Path]) -> Tuple[Path, np.ndarray, Dict[str, float]]:
    """Scan all matrices, return the one with the largest egress/ingress skew."""
    def score(s):
        return max(s["row_max_over_mean"], s["col_max_over_mean"])
    best = None
    for p in paths:
        D = load_matrix(p)
        s = matrix_skew(D)
        if best is None or score(s) > score(best[2]):
            best = (p, D, s)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices-dir", default=None,
                   help="Directory of per-layer demand-matrix .txt files. "
                        "Required unless --synth-uniform is set.")
    p.add_argument("--layer", default="most-skewed",
                   help="'most-skewed' (default), an integer layer index, or "
                        "a path to a single .txt matrix file.")
    p.add_argument("--synth-uniform", action="store_true",
                   help="Construct a uniform-A2A demand matrix instead of "
                        "loading a profiled one. Uses --N, --seqlen, "
                        "--hidden-size, --topk, --dtype-bytes.")
    p.add_argument("--N", type=int, default=64,
                   help="Number of ranks for synthetic uniform DM.")
    p.add_argument("--seqlen", type=int, default=8192,
                   help="Tokens per source rank for synthetic uniform DM.")
    p.add_argument("--hidden-size", type=int, default=4096,
                   help="Model hidden size (Qwen3-235B-A22B = 4096).")
    p.add_argument("--topk", type=int, default=8,
                   help="MoE top-k routing factor (Qwen3-235B-A22B = 8).")
    p.add_argument("--dtype-bytes", type=int, default=2,
                   help="Activation dtype size in bytes (bf16 = 2).")
    p.add_argument("--planes", default="1,2,4,8,16,32",
                   help="Comma-separated spectra plane counts.")
    p.add_argument("--reconfig-delay-us", type=float,
                   default=DEFAULT_RECONFIG_DELAY_US)
    p.add_argument("--wgs-per-gpu", type=int, default=DEFAULT_WGS_PER_GPU)
    p.add_argument("--wg-speed-gbps", type=float, default=DEFAULT_WG_SPEED_GBPS)
    p.add_argument("--nvlink-bw-GBs", type=float, default=DEFAULT_NVLINK_BW_GBs)
    p.add_argument("--ib-rail-bw-GBs", type=float, default=DEFAULT_IB_RAIL_BW_GBs)
    p.add_argument("--ib-num-rails", type=int, default=DEFAULT_IB_NUM_RAILS)
    p.add_argument("--ib-latency-us", type=float, default=DEFAULT_IB_LATENCY_US)
    p.add_argument("--logs", default="logs/dm_replay",
                   help="Output directory for results.json.")
    args = p.parse_args()

    # Build the demand matrix.
    if args.synth_uniform:
        D = make_uniform_dm(args.N, args.seqlen, args.hidden_size,
                            args.topk, args.dtype_bytes)
        path = Path(f"synth_uniform_seq{args.seqlen}.synth")
        print(f"Synthetic uniform DM: N={args.N}, seqlen={args.seqlen}, "
              f"hidden={args.hidden_size}, topk={args.topk}")
    else:
        if not args.matrices_dir:
            p.error("--matrices-dir is required unless --synth-uniform is set.")
        matrices_dir = Path(args.matrices_dir)
        if Path(args.layer).is_file():
            path = Path(args.layer)
            D = load_matrix(path)
        elif args.layer == "most-skewed":
            paths = discover_matrices(matrices_dir)
            print(f"Scanning {len(paths)} layers for the most-skewed matrix ...")
            path, D, _ = pick_most_skewed(paths)
            print(f"  picked: {path.name}")
        else:
            idx = int(args.layer)
            match = sorted(matrices_dir.glob(f"*layer{idx:02d}*.txt"))
            if not match:
                match = sorted(matrices_dir.glob(f"*layer{idx}*.txt"))
            if not match:
                raise FileNotFoundError(f"No matrix file for layer {idx} in {matrices_dir}")
            path = match[0]
            D = load_matrix(path)

    skew = matrix_skew(D)
    N = D.shape[0]
    print(f"\nDemand matrix: {path.name}")
    print(f"  size: {N}×{N}")
    print(f"  total: {skew['total_MB']:.1f} MB")
    print(f"  row_max/mean: {skew['row_max_over_mean']:.3f}  (egress skew)")
    print(f"  col_max/mean: {skew['col_max_over_mean']:.3f}  (ingress skew — hot experts)")
    print(f"  pair_max/mean: {skew['pair_max_over_mean']:.3f}")
    print(f"  CV (nonzero entries): {skew['cv']:.3f}")

    planes = [int(x) for x in args.planes.split(",") if x.strip()]

    # Run each fabric.
    print("\nReplaying through each fabric (single layer):")
    results: List[FabricResult] = []
    results.append(replay_fabric(
        "nvlink",
        [D],
        lambda D: nvlink_time_us(D, args.nvlink_bw_GBs),
    ))
    results.append(replay_fabric(
        "ib",
        [D],
        lambda D: ib_time_us(D, args.ib_rail_bw_GBs, args.ib_num_rails,
                             args.ib_latency_us),
    ))
    for s in planes:
        results.append(replay_fabric(
            f"spectra s={s}",
            [D],
            lambda D, s=s: spectra_time_us(
                D, num_planes=s,
                wgs_per_gpu=args.wgs_per_gpu,
                wg_speed_gbps=args.wg_speed_gbps,
                delta_us=args.reconfig_delay_us,
            ),
        ))

    # Persist.
    log_dir = Path(args.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "matrices_dir": str(args.matrices_dir) if args.matrices_dir else None,
            "synth_uniform": args.synth_uniform,
            "seqlen": args.seqlen if args.synth_uniform else None,
            "layer_file": path.name,
            "matrix_size": N,
            "skew": skew,
            "planes": planes,
            "reconfig_delay_us": args.reconfig_delay_us,
            "wgs_per_gpu": args.wgs_per_gpu,
            "wg_speed_gbps": args.wg_speed_gbps,
            "nvlink_bw_GBs": args.nvlink_bw_GBs,
            "ib_rail_bw_GBs": args.ib_rail_bw_GBs,
            "ib_latency_us": args.ib_latency_us,
        },
        "results": [
            {
                "label": r.label.replace("nvlink", "nvlink (1x64)").replace("ib", "ib (8x8)"),
                "total_sec": r.total_us / 1e6,
                "breakdown": {"AllToAll": r.total_us / 1e6},
                "per_layer_us": r.per_layer_us,
            }
            for r in results
        ],
    }
    out_path = log_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
