"""Fabric sweep: NVL64 vs 8x8 IB vs Spectra (plane-count sweep).

Runs the three interconnect modes on the same DeepSeek-V3 / EP=64 / prompt=16K
training-style workload, captures each run's stdout, parses the APEX
"Time breakdown" table, and prints a comm-only comparison plus the spectra
plane-count curve.

For spectra, also runs a paired delta=0 pass per plane count so the chart can
break the bar into (actual comm) + (reconfig overhead). The sweep emits a JSON
result file consumed by plot_fabric_sweep.py.

Usage:
  uv run python run_fabric_sweep.py
  uv run python run_fabric_sweep.py --planes 1,2,4,8,16,32 --logs logs/
  uv run python plot_fabric_sweep.py --input logs/fabric_sweep/results.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Comm names that come from CommType.name plus the cross-stage SendRecv.
COMM_NAMES = {
    "AllReduce",
    "AllGather",
    "ReduceScatter",
    "AllToAll",
    "SendRecv",
}

# Common APEX cell names (kept for compute-side reporting).
COMPUTE_NAMES = {
    "MHA", "MQA", "BiMHA",
    "MLPFilter", "GLUFilter", "SwiGLUFilter",
    "MoE", "SwiMoE",
    "Idle",
}

HERE = Path(__file__).resolve().parent


@dataclass
class RunSpec:
    label: str
    args: List[str]


@dataclass
class RunResult:
    label: str
    total_sec: float
    breakdown: Dict[str, float]   # name -> seconds
    raw_log: str


def build_runs(model: str, gpus: int, prompt: int, output: int,
               reqs: int, ep: int, freq: int,
               planes: List[int],
               reconfig_delay_us: float,
               num_layers: Optional[int],
               max_batch_size: Optional[int]) -> List[RunSpec]:
    common = [
        "--model", model,
        "--gpu", "H100-SXM-80GB",
        "--prompt-len", str(prompt),
        "--output-len", str(output),
        "--num-requests", str(reqs),
        "--force-ep", str(ep),
        "--frequency", str(freq),
    ]
    if num_layers is not None:
        common += ["--num-layers-override", str(num_layers)]
    if max_batch_size is not None:
        common += ["--max-batch-size", str(max_batch_size)]
    runs: List[RunSpec] = []
    runs.append(RunSpec(
        label="nvlink (1x64)",
        args=common + ["--num-nodes", "1", "--num-gpus-per-node", str(gpus),
                       "--interconnect", "nvlink"],
    ))
    runs.append(RunSpec(
        label="ib (8x8)",
        args=common + ["--num-nodes", "8", "--num-gpus-per-node", "8",
                       "--interconnect", "ib"],
    ))
    for p in planes:
        # Paired runs: with the configured reconfig delay (default), and with
        # delta=0 to isolate pure-comm time. The diff is reconfig overhead.
        runs.append(RunSpec(
            label=f"spectra s={p}",
            args=common + ["--num-nodes", "1", "--num-gpus-per-node", str(gpus),
                           "--interconnect", "spectra", "--num-planes", str(p),
                           "--reconfig-delay-us", str(reconfig_delay_us)],
        ))
        # δ=0 workaround: use a tiny but nonzero delta to avoid an apparent
        # degenerate path in the Julia solver at exactly delta=0.
        runs.append(RunSpec(
            label=f"spectra s={p} d=0",
            args=common + ["--num-nodes", "1", "--num-gpus-per-node", str(gpus),
                           "--interconnect", "spectra", "--num-planes", str(p),
                           "--reconfig-delay-us", "0.0001"],
        ))
    return runs


# PrettyTable config in engine.py uses border=True / vrules=NONE, so rows look
# like (leading 2 spaces, columns separated by 2+ spaces, "±" optional):
#     MQA        59.11 ± 0.0   44.7
#     Total      132.18        100.0
# Border lines are runs of "-".
_ROW_RE = re.compile(
    r"^(?P<name>\S(?:.*?\S)?)\s{2,}(?P<time>[0-9.]+)(?:\s*±\s*[0-9.]+)?\s{2,}(?P<ratio>[0-9.]+)\s*$"
)


def parse_breakdown(stdout: str) -> Tuple[float, Dict[str, float]]:
    """Return (total_sec, {name: seconds}) from the LAST breakdown table found."""
    breakdown: Dict[str, float] = {}
    total = 0.0

    last_idx = stdout.rfind("* Time breakdown:")
    if last_idx == -1:
        return 0.0, {}
    tail = stdout[last_idx:].splitlines()

    for raw in tail:
        line = raw.strip()
        if not line or set(line) <= {"-"}:
            continue
        if line.startswith("*"):
            continue
        m = _ROW_RE.match(line)
        if m is None:
            continue
        name = m.group("name").strip()
        if name in ("Name", "Energy Consumption:"):
            continue
        try:
            t = float(m.group("time"))
        except ValueError:
            continue
        if name == "Total":
            total = t
            break
        breakdown[name] = breakdown.get(name, 0.0) + t
    return total, breakdown


_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def run_one(spec: RunSpec, log_dir: Optional[Path]) -> RunResult:
    cmd = ["uv", "run", "python", "main.py"] + spec.args
    t0 = time.time()
    _log(f"  [start] {spec.label}")
    proc = subprocess.run(
        cmd,
        cwd=str(HERE),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    log = proc.stdout + "\n[STDERR]\n" + proc.stderr
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.label)
        (log_dir / f"{slug}.log").write_text(log)
    if proc.returncode != 0:
        _log(f"  [FAIL ] {spec.label}  ({elapsed:.0f}s, exit {proc.returncode})")
        for line in proc.stderr.strip().splitlines()[-10:]:
            _log("    " + line)
        return RunResult(spec.label, 0.0, {}, log)
    total, breakdown = parse_breakdown(proc.stdout)
    if total == 0.0:
        _log(f"  [WARN ] {spec.label}  ({elapsed:.0f}s, no breakdown)")
    else:
        a2a = breakdown.get("AllToAll", 0.0)
        _log(f"  [done ] {spec.label}  ({elapsed:.0f}s, total={total:.2f}s, A2A={a2a:.3f}s)")
    return RunResult(spec.label, total, breakdown, log)


def comm_total(r: RunResult) -> float:
    return sum(v for k, v in r.breakdown.items() if k in COMM_NAMES)


def fmt_row(label: str, vals: List[str], widths: List[int]) -> str:
    cells = [label.ljust(widths[0])]
    for v, w in zip(vals, widths[1:]):
        cells.append(v.rjust(w))
    return "  " + "  ".join(cells)


def print_summary(results: List[RunResult]) -> None:
    # Comm-type union (in stable order).
    order = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll", "SendRecv"]
    seen = set()
    cols: List[str] = []
    for c in order:
        if any(c in r.breakdown for r in results):
            cols.append(c)
            seen.add(c)
    for r in results:
        for k in r.breakdown:
            if k in COMM_NAMES and k not in seen:
                cols.append(k)
                seen.add(k)

    headers = ["Fabric"] + cols + ["Comm total", "End-to-end", "Comm/Total"]
    widths = [max(len(h), 18) for h in headers]
    for r in results:
        widths[0] = max(widths[0], len(r.label))

    print()
    print("=" * 100)
    print("Communication time breakdown (seconds)")
    print("=" * 100)
    print(fmt_row(headers[0], headers[1:], widths))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in results:
        vals: List[str] = []
        for c in cols:
            v = r.breakdown.get(c, 0.0)
            vals.append(f"{v:.3f}" if v > 0 else "—")
        ctot = comm_total(r)
        vals.append(f"{ctot:.3f}")
        vals.append(f"{r.total_sec:.3f}")
        ratio = (ctot / r.total_sec * 100.0) if r.total_sec > 0 else 0.0
        vals.append(f"{ratio:.1f}%")
        print(fmt_row(r.label, vals, widths))
    print()


def print_spectra_curve(results: List[RunResult]) -> None:
    sp = [(int(re.search(r"s=(\d+)", r.label).group(1)), r)
          for r in results if r.label.startswith("spectra s=")]
    if not sp:
        return
    sp.sort()
    print("Spectra plane-count curve")
    print("-" * 60)
    print(f"  {'planes':>7}  {'comm (s)':>10}  {'a2a (s)':>10}  {'ar (s)':>10}  {'total (s)':>10}")
    for p, r in sp:
        c = comm_total(r)
        a2a = r.breakdown.get("AllToAll", 0.0)
        ar = r.breakdown.get("AllReduce", 0.0)
        print(f"  {p:>7}  {c:>10.3f}  {a2a:>10.3f}  {ar:>10.3f}  {r.total_sec:>10.3f}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-v3")
    p.add_argument("--gpus", type=int, default=64)
    p.add_argument("--prompt", type=int, default=16384)
    p.add_argument("--output", type=int, default=1)
    p.add_argument("--reqs", type=int, default=4096)
    p.add_argument("--ep", type=int, default=64)
    p.add_argument("--freq", type=int, default=1980)
    p.add_argument("--planes", default="1,2,4,8,16",
                   help="Comma-separated spectra plane counts.")
    p.add_argument("--reconfig-delay-us", type=float, default=10.0,
                   help="Reconfig delay (µs) for spectra non-zero pass.")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Override num_hidden_layers (e.g. 1 for fast sweeps).")
    p.add_argument("--max-batch-size", type=int, default=None,
                   help="Force max_batch_size (e.g. 1 to fix per-iter token count).")
    p.add_argument("--parallel", type=int, default=1,
                   help="Run up to N configs concurrently (each spawns its own Julia).")
    p.add_argument("--logs", default="logs/fabric_sweep",
                   help="Directory to write per-run stdout logs and results.json.")
    p.add_argument("--skip-nvlink", action="store_true")
    p.add_argument("--skip-ib", action="store_true")
    p.add_argument("--with-delta-zero", action="store_true",
                   help="Include delta=0 paired runs for reconfig breakdown "
                        "(disabled by default — Julia solver has degenerate "
                        "behavior at delta=0).")
    args = p.parse_args()

    planes = [int(x) for x in args.planes.split(",") if x.strip()]
    runs = build_runs(args.model, args.gpus, args.prompt, args.output,
                      args.reqs, args.ep, args.freq, planes,
                      args.reconfig_delay_us, args.num_layers,
                      args.max_batch_size)
    if args.skip_nvlink:
        runs = [r for r in runs if not r.label.startswith("nvlink")]
    if args.skip_ib:
        runs = [r for r in runs if not r.label.startswith("ib")]
    if not args.with_delta_zero:
        runs = [r for r in runs if not r.label.endswith("d=0")]

    log_dir = HERE / args.logs if args.logs else None

    results: List[RunResult] = []
    if args.parallel <= 1:
        for spec in runs:
            results.append(run_one(spec, log_dir))
    else:
        _log(f"Launching {len(runs)} configs with parallelism={args.parallel}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            future_to_spec = {ex.submit(run_one, spec, log_dir): spec for spec in runs}
            results_by_label: Dict[str, RunResult] = {}
            for fut in concurrent.futures.as_completed(future_to_spec):
                r = fut.result()
                results_by_label[r.label] = r
        # Re-order to match the original spec order so prints are stable.
        results = [results_by_label[s.label] for s in runs]

    print_summary(results)
    print_spectra_curve(results)

    if log_dir:
        out = {
            "config": {
                "model": args.model, "gpus": args.gpus, "prompt": args.prompt,
                "output": args.output, "reqs": args.reqs, "ep": args.ep,
                "freq": args.freq, "planes": planes,
                "reconfig_delay_us": args.reconfig_delay_us,
            },
            "results": [
                {"label": r.label, "total_sec": r.total_sec,
                 "breakdown": r.breakdown}
                for r in results
            ],
        }
        (log_dir / "results.json").write_text(json.dumps(out, indent=2))
        print(f"Per-run logs + results.json: {log_dir}")


if __name__ == "__main__":
    sys.exit(main())
