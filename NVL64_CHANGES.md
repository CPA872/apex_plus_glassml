# NVL64 Simulation Support

## Problem

APEX+ could only simulate up to 8 H100 GPUs on a single node. Communication timing is profile-driven — it looks up `(num_nodes, num_gpus_per_node)` in CSV files, and only `(1,2)`, `(1,4)`, `(1,8)` exist for H100. Any collective on >8 GPUs hit an assert failure in `comm_profile.py`.

## Solution

Added an analytical fallback in the communication profiling layer. When `get_comm_time` finds no matching rows for the requested GPU count, it extrapolates from the largest available profiled data (8-GPU NVLink) using standard collective communication formulas.

### Why this works for NVL64

- NVL64 is a single NVLink domain — per-GPU algorithmic bandwidth is roughly constant regardless of GPU count
- The `(N-1)/N` scaling factor only changes from 0.875 (N=8) to 0.984 (N=64)
- Latency grows logarithmically with GPU count (tree-based algorithms)

## Files Changed

### `apex_plus/simulator/comm_profile.py`

Added ~65 lines, no existing logic altered for profiled configurations.

**New functions:**

- `_bandwidth_factor(comm_type, n)` — Returns the multiplicative factor on `data_size / bandwidth` for each collective type:
  - AllReduce: `2*(N-1)/N`
  - AllGather, ReduceScatter, AllToAll: `(N-1)/N`

- `_extract_bandwidth_and_latency(comm_type, op_kind, gpu, dtype_str, ref_n=8)` — Extracts per-GPU algorithmic bandwidth and base latency from the reference N-GPU profiled data. Uses the two largest message sizes to derive bandwidth via linear regression. Falls back to smaller reference counts (4, 2) if the requested reference isn't available. Results are LRU-cached.

- `_analytical_comm_time(comm_type, gpu, num_nodes, num_gpus_per_node, dtype_str, size_kb)` — Computes estimated communication time as:
  ```
  time = latency * log2(target_N) / log2(ref_N) + bandwidth_factor(target_N) * size / algo_bw
  ```
  Emits a warning when `num_nodes > 1` since the model assumes NVLink, not InfiniBand.

**Modified function:**

- `get_comm_time()` — Replaced the `assert not df.empty` with a conditional: if profiled data exists, use interpolation (unchanged behavior); otherwise fall back to `_analytical_comm_time`.

### `apex_plus/simulator/simulator.py`

Two minor fixes for edge cases exposed by large GPU counts:

1. **KV token size zero-division** (line 349-352): With many GPUs, some execution plans assign devices only FFN tasks (no attention heads), resulting in `kv_token_size = 0`. Added a filter to skip these devices in the `max_num_tokens` calculation.

2. **Early-exit tuple length** (lines 643, 653): `sub_simulate` returns 7 values on success but the early-exit failure paths returned only 6 `None`s. Fixed to return 7 `None`s to match the caller's unpacking.

## Usage

```bash
# Single-node NVL64
python main.py --model llama3-70b --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --trace-file ./traces/llama/lmsys_05.jsonl

# Also works for arbitrary GPU counts (16, 32, etc.)
python main.py --model llama3-70b --num-nodes 1 --num-gpus-per-node 32 \
  --gpu H100-SXM-80GB --prompt-len 128 --output-len 2048
```

Existing configurations (1-8 GPUs) are unaffected — they still use the profiled CSV data path.

## Validation

Tested with LLaMA3-70B on lmsys trace (`traces/llama/lmsys_05.jsonl`):

| Metric | 8× H100 | 64× H100 (NVL64) | Ratio |
|---|---|---|---|
| Throughput (tokens gen/s) | 256 | 1,510 | 5.9× |
| Requests/sec | 1.58 | 9.32 | 5.9× |
| TTFT | 48.2 ms | 46.1 ms | ~same |
| TPOT | 27.6 ms | 27.6 ms | ~same |
| MFU | 5.0% | 29.5% | 5.9× |
| Optimal plan | TP8 (1 replica) | TP8 × DP4 × PP2 | — |

The optimizer correctly chooses TP8 as the intra-replica strategy (matching `num_kv_heads=8`) and scales throughput via data parallelism, which is the expected behavior.

## Limitations

- The analytical model assumes a full-bisection NVLink domain. Multi-node (InfiniBand/EFA) estimates will be inaccurate — a warning is emitted.
- For highest accuracy on a specific NVL64 system, run the profiling script (`profile/scripts/comm.py`) on the actual hardware to generate CSV data for the exact `(num_nodes, num_gpus_per_node)` configuration.
