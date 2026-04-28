# spectra_solver — Python API for OCS Scheduling

## What this does

Given a traffic demand matrix D (who needs to send how much data to whom), decompose it into a sequence of conflict-free switching configurations (permutation matrices) and their durations. Optionally schedule them across multiple parallel optical circuit switches.

Two algorithms are available:

- **SPECTRA** — sparsity-aware decomposition + parallel-plane scheduling. Produces fewer permutations (fewer reconfigurations), good for sparse/skewed traffic like MoE.
- **WWFA** — wavefront arbiter decomposition. Simpler, produces more permutations.

## Quick start

```python
from spectra.spectra_solver import spectra, wwfa
import numpy as np

# 4x4 demand matrix: D[i][j] = traffic from node i to node j
D = [[0.0, 0.3, 0.2, 0.1],
     [0.1, 0.0, 0.4, 0.1],
     [0.2, 0.1, 0.0, 0.3],
     [0.3, 0.2, 0.1, 0.0]]

# Decompose and schedule across 2 parallel switches, 0.01 reconfiguration delay
perms, durations, makespan = spectra(D, s=2, delta=0.01)

# perms:     list of k permutation matrices (numpy arrays, n×n, entries 0 or 1)
# durations: list of k floats (how long each permutation is held)
# makespan:  float (total completion time across all switches)
```

## API

### `spectra(D, s=4, delta=0.01)`

Decompose D into permutations and schedule them across s parallel OCS planes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `D` | array-like (n×n) | Demand matrix. Accepts list-of-lists or numpy array. |
| `s` | int | Number of parallel optical circuit switches. |
| `delta` | float | Reconfiguration delay (time cost to change a switch's permutation). |

**Returns** `(perms, durations, makespan)`:
- `perms` — list of k `np.ndarray` of shape (n, n), dtype int8, entries in {0, 1}
- `durations` — list of k floats, sorted descending
- `makespan` — float

### `wwfa(D)`

Decompose D using the wavefront arbiter (no scheduling step).

| Parameter | Type | Description |
|-----------|------|-------------|
| `D` | array-like (n×n) | Demand matrix. |

**Returns** `(perms, durations)`:
- `perms` — list of k `np.ndarray` of shape (n, n), dtype int8
- `durations` — list of k floats, sorted descending

## Output interpretation

Each permutation matrix P is a conflict-free 1-to-1 assignment: `P[i][j] == 1` means node i sends to node j during that configuration. Every row and column of P sums to exactly 1.

The weighted sum of all permutations covers the demand:

```
sum(durations[i] * perms[i] for i in range(k))  >=  D   (elementwise)
```

For SPECTRA with s parallel switches, the permutations are distributed across s switches. The makespan is the time until the slowest switch finishes (not the sum of all durations).

## Performance

First call includes Julia JIT warmup (~2-3s). Subsequent calls are fast:

| Matrix size | SPECTRA time | WWFA time | SPECTRA k | WWFA k |
|-------------|-------------|-----------|-----------|--------|
| 8×8 | <1ms | <1ms | 8 | 64 |
| 32×32 | ~1.5ms | ~8ms | 32 | 1024 |
| 64×64 | ~13ms | ~350ms | 64 | 4096 |
| 72×72 | ~16ms | ~400ms | 72 | 5184 |

SPECTRA produces ~n permutations; WWFA produces ~n² permutations.

## Dependencies

- Python: `juliacall`, `numpy`
- Julia 1.12+: `DataStructures`, `Hungarian` (installed in `.venv/julia_env`)

The Julia runtime is loaded in-process on first call — no subprocess or JSON serialization.

## Files

| File | Purpose |
|------|---------|
| `spectra_solver.py` | Python module (this API) |
| `spectra_core.jl` | Lean Julia wrapper — core algorithms only, no PyCall |
| `spectra.jl` | Original full Julia implementation (not used by this API) |
| `tests/` | 54 reference test cases with saved inputs and outputs |
| `generate_test_cases.jl` | Regenerate reference test data |
| `verify_against_reference.jl` | Verify correctness after code changes |

## Testing

Generate reference outputs (once):
```bash
julia generate_test_cases.jl
```

Verify correctness after changes:
```bash
julia verify_against_reference.jl
```

Or from Python:
```python
from spectra.spectra_solver import spectra
import numpy as np

D = np.random.rand(16, 16)
D /= max(D.sum(axis=0).max(), D.sum(axis=1).max())

perms, durations, makespan = spectra(D, s=4, delta=0.01)

# Verify coverage
reconstructed = sum(w * p for w, p in zip(durations, perms))
assert np.all(reconstructed >= D - 1e-6)

# Verify permutations
for p in perms:
    assert np.all(p.sum(axis=0) == 1)
    assert np.all(p.sum(axis=1) == 1)
```
