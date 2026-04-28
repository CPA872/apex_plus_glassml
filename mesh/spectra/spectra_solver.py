"""
spectra_solver — Python interface to SPECTRA and WWFA scheduling algorithms.

Usage:
    from spectra.spectra_solver import spectra, wwfa

    # SPECTRA: decompose + schedule across s parallel OCS planes
    perms, durations, makespan = spectra(D, s=4, delta=0.01)

    # WWFA: wavefront arbiter decomposition only
    perms, durations = wwfa(D)

Input:
    D: n×n demand matrix (list of lists or numpy array)
    s: number of parallel OCS planes (SPECTRA only)
    delta: reconfiguration delay (SPECTRA only)

Output:
    perms: list of k permutation matrices, each n×n numpy array of 0/1
    durations: list of k floats (weight/duration for each permutation)
    makespan: float, completion time across all planes (SPECTRA only)
"""

import os
import numpy as np

_jl = None
_initialized = False

JULIA_EXE = os.environ.get(
    "PYTHON_JULIAPKG_EXE",
    os.path.expanduser("~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia"),
)
JULIA_PROJECT = os.environ.get(
    "JULIA_PROJECT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".venv", "julia_env"),
)
CORE_JL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectra_core.jl")


def _init():
    global _jl, _initialized
    if _initialized:
        return

    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", JULIA_EXE)
    os.environ.setdefault("JULIA_PROJECT", JULIA_PROJECT)

    import juliacall
    _jl = juliacall.Main
    _jl.seval(f'include("{CORE_JL}")')
    _initialized = True


def _to_julia_matrix(D):
    """Convert Python array to Julia Matrix{Float64}."""
    import juliacall
    D = np.asarray(D, dtype=np.float64)
    return juliacall.convert(_jl.Matrix[_jl.Float64], D)


def _perms_to_numpy(P_jl):
    """Convert Julia Vector{Matrix{Int16}} to list of numpy arrays."""
    return [np.array(p, dtype=np.int8) for p in P_jl]


def spectra(D, s=4, delta=0.01):
    """
    Decompose demand matrix D and schedule across s parallel OCS planes.

    Parameters
    ----------
    D : array-like, shape (n, n)
        Demand matrix. D[i,j] = traffic from source i to destination j.
    s : int
        Number of parallel optical circuit switches (planes).
    delta : float
        Reconfiguration delay per switch permutation change.

    Returns
    -------
    perms : list of np.ndarray
        k permutation matrices, each (n, n) with entries in {0, 1}.
    durations : list of float
        Weight (duration) for each permutation. Sorted descending.
    makespan : float
        Completion time (max load across the s planes).
    """
    _init()
    D_jl = _to_julia_matrix(D)
    P_jl, w_jl = _jl.SPECTRA(D_jl)
    makespan, _schedule = _jl.alg3(w_jl, s=s, delta=delta)
    perms = _perms_to_numpy(P_jl)
    durations = list(w_jl)
    return perms, durations, float(makespan)


def wwfa(D):
    """
    Decompose demand matrix D using the Wavefront Arbiter algorithm.

    Parameters
    ----------
    D : array-like, shape (n, n)
        Demand matrix. D[i,j] = traffic from source i to destination j.

    Returns
    -------
    perms : list of np.ndarray
        k permutation matrices, each (n, n) with entries in {0, 1}.
    durations : list of float
        Weight (duration) for each permutation. Sorted descending.
    """
    _init()
    D_jl = _to_julia_matrix(D)
    P_jl, w_jl = _jl.wwfa(D_jl)
    perms = _perms_to_numpy(P_jl)
    durations = list(w_jl)
    return perms, durations
