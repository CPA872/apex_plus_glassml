"""Clean wrapper around the Julia-backed Spectra solver.

The single public entry point is `spectra_simulate(D, num_planes, config)`,
which takes a demand matrix in bytes, the number of OCS planes, and a global
SpectraConfig, and returns the scheduled fabric latency in microseconds.

This is the only module that touches mesh/spectra/spectra_solver.py
(juliacall). Callers see a plain (D, s, config) -> float function.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from apex_plus.simulator.comm_profile import SpectraConfig

_SOLVER = None


def _get_solver():
    """Lazily import the Julia-backed solver. Cached after first call."""
    global _SOLVER
    if _SOLVER is not None:
        return _SOLVER

    # The solver lives at apex_plus/mesh/spectra/spectra_solver.py — sibling
    # of the apex_plus package, not on sys.path by default.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    spectra_dir = os.path.join(repo_root, "mesh", "spectra")
    if spectra_dir not in sys.path:
        sys.path.insert(0, spectra_dir)

    import spectra_solver  # type: ignore
    _SOLVER = spectra_solver
    return _SOLVER


def spectra_simulate(
    D: np.ndarray,
    num_planes: int,
    config: SpectraConfig,
) -> float:
    """Return scheduled fabric latency in microseconds for one demand matrix.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
        Demand matrix in bytes. D[i, j] = bytes from rank i to rank j.
    num_planes : int
        Number of parallel OCS planes (s).
    config : SpectraConfig
        Global Spectra configuration. Provides bandwidth-per-plane (for
        bytes→µs conversion) and reconfig delay (already in µs).

    Returns
    -------
    float
        Makespan in microseconds.
    """
    if D.size == 0 or float(D.sum()) == 0.0:
        return 0.0
    if num_planes <= 0:
        raise ValueError(f"num_planes must be positive, got {num_planes}")

    # Convert bytes -> microseconds by dividing through per-port bandwidth.
    # The SPECTRA solver schedules permutations across s planes; at any moment
    # each plane realizes a 1-to-1 matching, so the rate at which one
    # demand-matrix entry drains is the per-(GPU, plane) port speed.
    # Treating the weights it schedules as microseconds means the returned
    # makespan is in µs.
    bw = config.per_port_bytes_per_us
    if bw <= 0:
        raise ValueError(
            f"SpectraConfig per-port bandwidth must be positive, got "
            f"wgs_per_gpu={config.wgs_per_gpu}, num_planes={config.num_planes}, "
            f"wg_speed_gbps={config.wg_speed_gbps}"
        )

    D_us = np.asarray(D, dtype=np.float64) / bw

    solver = _get_solver()
    _perms, _durations, makespan = solver.spectra(
        D_us, s=num_planes, delta=config.reconfig_delay_us
    )
    return float(makespan)
