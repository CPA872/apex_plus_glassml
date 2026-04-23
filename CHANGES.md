# APEX+ Simulation Extensions: NVL64 & IB Inter-Node Communication

## Overview

These changes extend APEX+ to simulate large GPU configurations beyond the original 8-GPU-per-node limit. The major capabilities added are:

1. **Analytical NVLink fallback** for single-node configurations with >8 GPUs (e.g., NVL64/NVL72)
2. **Hierarchical IB inter-node communication model** for multi-node clusters (e.g., 8 nodes x 8 H100 GPUs)
3. **Training-style parallelism constraint** (`--force-ep`) that enforces TP x EP = total GPUs
4. **GEMM profile extrapolation** for token counts beyond the profiled range
5. **Demand matrix extraction** (`--demand-matrix`) that outputs R×R byte-traffic matrices per collective from execution plans

6. **Communication energy modeling** using pJ/bit metrics for NVLink and InfiniBand, with a top-level YAML config file

Additionally, several bugs were fixed (including two pre-existing ones in Mixtral IR and the model registry) and new model configs were added for evaluation.

---

## Files Changed

### 1. `apex_plus/simulator/comm_profile.py` — Core communication modeling (~220 lines added)

**Original behavior:** Communication time was looked up from profiled CSV files. Only `(num_nodes=1, num_gpus_per_node=2|4|8)` entries existed for H100. Any unprofiled configuration hit an assert failure.

#### New: `InterconnectConfig` dataclass
```python
@dataclass(frozen=True)
class InterconnectConfig:
    mode: str = "nvlink"            # "nvlink" or "ib"
    ib_bw_per_rail_GBs: float = 50.0   # 400Gbps = 50 GB/s per rail
    num_rails: int = 8              # DGX H100 has 8 ConnectX-7 ports
    ib_latency_us: float = 3.0     # base IB RDMA latency per message
```
Frozen (hashable) so it works with existing `@lru_cache` decorators.

#### New: Analytical NVLink model (for NVL64-style single-node)

Three new functions implement the analytical fallback for unprofiled GPU counts within a single NVLink domain:

- **`_bandwidth_factor(comm_type, n)`** — Returns the algorithm-theoretic data volume factor:
  - AllReduce: `2*(N-1)/N`
  - AllGather, ReduceScatter, AllToAll: `(N-1)/N`

- **`_extract_bandwidth_and_latency(comm_type, op_kind, gpu, dtype_str, ref_n=8)`** — Extracts per-GPU algorithmic bandwidth and base latency from the largest-available profiled reference data (8-GPU by default). Uses linear regression on the two largest message sizes. LRU-cached.

- **`_analytical_comm_time(comm_type, gpu, num_nodes, num_gpus_per_node, dtype_str, size_kb)`** — Computes:
  ```
  time = latency + bandwidth_factor(target_N) * size / algo_bw
  ```
  **Latency model:** For single-node (NVSwitch), latency is **constant** (non-blocking crossbar, single hop regardless of GPU count). For multi-node, latency scales as `log2(target_N) / log2(ref_N)`. This distinction is critical: NVSwitch-based systems like NVL64/NVL72 provide full-bisection bandwidth with constant-latency access to any GPU, unlike tree/ring topologies.

#### New: Hierarchical IB model (for multi-node clusters)

Two new functions decompose multi-node collectives into NVLink + IB phases:

- **`_ib_phase_time(comm_type, num_nodes, data_bytes_per_rail, config)`** — Computes the inter-node IB phase time. AllToAll uses `latency * (N-1)` (point-to-point exchanges); others use `latency * ceil(log2(N))` (tree/ring).

- **`_hierarchical_comm_time(comm_type, gpu, num_nodes, num_gpus_per_node, dtype, num_elements, config)`** — Decomposes each collective into sequential phases:

  | Collective | Phase 1 (NVLink) | Phase 2 (IB) | Phase 3 (NVLink) |
  |---|---|---|---|
  | **AllReduce** | ReduceScatter | AllReduce | AllGather |
  | **AllGather** | AllGather | AllGather | — |
  | **ReduceScatter** | — | ReduceScatter | ReduceScatter |
  | **AllToAll** | AllToAll | AllToAll | — |

  Intra-node phases reuse the profiled/analytical NVLink model (`get_comm_time(..., num_nodes=1)`). IB data volume is divided by `num_rails` (multi-rail parallelism).

#### Modified: `get_comm_time()` dispatch logic

- Added `interconnect: InterconnectConfig = None` parameter
- Changed guard from `num_nodes == 1 and num_gpus_per_node == 1` to `num_nodes * num_gpus_per_node <= 1` (fixes ZeroDivisionError when `num_devices=0` reached `_bandwidth_factor`)
- Dispatch order:
  1. `num_nodes > 1` and `interconnect.mode == "ib"` → `_hierarchical_comm_time()`
  2. Profiled CSV data exists → `_interpolate()` (unchanged)
  3. Fallback → `_analytical_comm_time()` (NVLink domain)
- Replaced `assert not df.empty` with conditional fallback (non-breaking for existing configs)

#### Modified: `get_p2p_comm_time()` for cross-node pipeline parallelism

- Added `interconnect` parameter
- When `num_nodes >= 2` and mode is `"ib"`: `time = ib_latency + data_bytes / ib_bw_per_rail` (single-rail P2P)

---

### 2. `apex_plus/simulator/comp_profile.py` — GEMM profile extrapolation (~10 lines added)

**Problem:** The GEMM compute profile is only measured for token counts up to `n = 65536` (`MAX_NUM_INPUT_TOKENS`). When cell-level data parallelism concentrates tokens from many attention replicas into a single MoE cell replica (e.g., 64 attention replicas × 8192 tokens each → 131072 per-expert tokens at the MoE cell), the GEMM lookup fails with an assertion error.

**Fix:** Added `MAX_NUM_INPUT_TOKENS` constant and linear extrapolation in `gemm_time()`. When `n > MAX_NUM_INPUT_TOKENS`, the function queries the profile at the maximum profiled n and scales linearly:
```python
if n > max_profiled_n:
    scale = n / max_profiled_n
    time, energy = _gemm_time(gpu, frequency, m, k, max_profiled_n, dtype)
    return time * scale, energy * scale
```

**Justification:** For large n, GEMMs are compute-bound and execution time scales linearly with n. This approximation is accurate for the batch sizes encountered in training-style simulations.

---

### 3. `apex_plus/simulator/simulator.py` — Threading interconnect config, bug fixes (~20 lines changed)

- `Simulator.__init__`: accepts and stores `self.interconnect`
- `get_stage_execution_time()` (line ~877): passes `self.interconnect` to `get_comm_time()` for reshard collectives
- `get_cross_stage_comm_time()` (line ~907, ~916): passes `self.interconnect` to both `get_p2p_comm_time()` calls

**Bug fixes:**

1. **KV token size ZeroDivisionError** (line ~350): With large GPU counts, some execution plans assign devices only FFN tasks (no attention heads), resulting in `kv_token_sizes[i] == 0`. Added `if kv_token_sizes[i] > 0` filter in `max_num_tokens` calculation.

2. **Early-exit tuple mismatch** (lines ~642, ~652): `sub_simulate` returns 7 values on success but early-exit failure paths returned 6 `None`s. Fixed to return 7 `None`s.

3. **Batch size check for mixed cell-DP** (line ~620): The original check `num_tokens * num_attn_cell_replicas / min_num_replicas > MAX_NUM_INPUT_TOKENS` compared per-cell-replica tokens, but the GEMM profile's n dimension corresponds to per-device tokens. With cell-DP64 for attention (64 replicas × 1 device) and EP64 for MoE (1 replica × 64 devices), the old check computed `8192 * 64 / 1 = 524K`, exceeding the 65536 limit, preventing even a single request from being batched. Fixed to `num_tokens > MAX_NUM_INPUT_TOKENS` — the per-attention-replica token count, which is the actual GEMM n dimension.

---

### 4. `apex_plus/search/engine.py` — Threading interconnect + force-EP + robustness fix (~30 lines changed)

- `SearchEngine.__init__`: accepts `interconnect=None`, passes to `Simulator()`
- `search()`: Changed `requests, output = self.simulator.simulate(...)` to first capture the result and check for `None` before unpacking. This handles `simulate()` returning bare `None` (OOM at line 339) vs `(None, None)` (sub_simulate failure) vs `(requests, SimulatorOutput)` (success).

#### New: `--force-ep` plan filtering with training constraint

Added `force_ep: int = 0` parameter to `search()`. When set, filters candidate plans to enforce training-style parallelism where **TP × EP = total GPUs**:

```python
if force_ep > 0:
    filtered = []
    for plan in candidate_plans:
        stage_sched = plan.parallel_schedule.stage_schedule
        # Find MoE EP degree.
        moe_ep = None
        for cs in stage_sched.cell_schedules:
            if cs.cell.get_name() in ("MoE", "SwiMoE"):
                moe_ep = cs.get_num_devices()
                break
        if moe_ep is None or moe_ep < force_ep:
            continue
        # Enforce training constraint: attention cell-DP == MoE EP.
        # This means TP = stage_devices / EP.
        attn_ok = True
        for cs in stage_sched.cell_schedules:
            if cs.cell.is_attn() and cs.num_replicas != moe_ep:
                attn_ok = False
                break
        if attn_ok:
            filtered.append(plan)
    candidate_plans = filtered
```

**Why this matters:** APEX+ uses per-cell parallelism where different cells can have different strategies (e.g., TP8 for attention, EP64 for MoE with resharding between them). This is more general than training systems, where TP and EP share the same device group. Without the training constraint, `--force-ep 64` on 64 GPUs produces TP8 × EP64 with costly ReduceScatter(8) + AllGather(8) resharding operations that don't exist in real training.

With the constraint:
- Attention cell-DP = EP (e.g., 64 replicas × 1 device = TP1)
- MoE EP = force_ep (e.g., 1 replica × 64 devices = EP64)
- TP = total / EP = 64 / 64 = 1
- Resharding ops become 1-device no-ops (no communication)

---

### 5. `apex_plus/simulator/demand_matrix.py` — Demand matrix extraction (NEW, ~200 lines)

Extracts R×R byte-traffic matrices from execution plans, where R = total GPU ranks and `matrix[i][j]` = bytes sent from rank i to rank j per collective call.

**Key functions:**

- **`get_device_ids(cluster)`** — Recursively extracts GPU `device_id` values from the Cluster hierarchy.

- **`_zipf_load_ratios(num_experts, num_devices, zipf_s)`** — Computes per-GPU load ratios under Zipfian expert popularity. Returns array where `ratio[g] = gpu_g_load / avg_load`. Uses the same round-robin expert assignment as the simulator's `_zipf_skew()`, but returns per-destination ratios instead of a scalar max.

- **`extract_demand_matrices(plan, cluster, model, act_dtype, num_tokens, moe_skew)`** — Main function. Walks the execution plan structure, mirrors the simulator's volume computation (`get_stage_execution_time` lines 782-922), and fills per-rank-pair matrices.

- **`save_demand_matrices(matrices, output_dir, prefix)`** — Writes matrices as `repr(list-of-lists)` text files, matching the `demand-matrix/` project format.

**How each parallelism type is handled:**

| Parallelism | Collective | Matrix Semantics |
|---|---|---|
| **TP** | AllReduce | `matrix[i][j] = tensor_bytes` for all peers in TP group |
| **EP** | AllToAll | `matrix[i][j] = per_peer_bytes * load_ratio[dst]` (Zipf-skewed) |
| **DP** | None (inference) | Block-diagonal structure: zero cross-replica traffic |
| **Resharding** | ReduceScatter/AllGather | `matrix[i][j] = shard_bytes` for all peers in group |

**EP with Zipf skew produces asymmetric matrices:** All source ranks send the same total distribution, but destination ranks receive proportional to their expert popularity. GPU 0 (hottest experts via round-robin assignment) receives up to 31× more than the coldest GPU at Zipf(1.0) with 256 experts across 64 GPUs.

### 6. Communication energy modeling — `comm_profile.py`, `simulator.py`, `energy_config.yaml` (~50 lines added)

**Problem:** APEX+ modeled compute energy from profiled CSVs (GEMM, MHA, MLP energy at specific GPU frequencies in μJ) but had **zero communication energy**. In `get_stage_execution_time()`, resharding comms appended time to `execution_time` but nothing to `execution_energy`. All reported energy was purely from GPU compute, understating total system energy for communication-heavy workloads like MoE AllToAll.

#### New: `energy_config.yaml` (top-level config file)

```yaml
nvlink_intra_node_pj_per_bit: 1.3    # <=8 GPUs/node (DGX-internal)
nvlink_rack_scale_pj_per_bit: 5.0    # >8 GPUs/node (NVL72 copper cables)
ib_pj_per_bit: 70.0                  # Full IB path (NIC + optics + switch)
```

Three distinct energy rates handle the physical differences between interconnect paths:

| Link Type | Value | Physical Basis | Source |
|---|---|---|---|
| Intra-node NVLink | 1.3 pJ/bit | Short PCB traces, GRS signaling (≤8 GPUs, DGX-internal) | Wei et al., "NVLink-C2C," **ISSCC 2023**, Paper 9.3; Poulton et al., **IEEE JSSC** vol. 54, 2019 (1.17 pJ/bit GRS SerDes) |
| Rack-scale NVLink | 5.0 pJ/bit | 112G PAM4 over copper cables through NVSwitch trays (NVL72) | Estimated from PCIe Gen5 class (~6.5 pJ/bit implied by NVIDIA's "5x more efficient" C2C claim; Bichan et al., **IEEE CICC 2020**: 11.4 pJ/bit measured) |
| IB full path | 70.0 pJ/bit | NIC (~30) + 400G optics (~25) + switch ASIC (~12) | NIC: NVIDIA CX-7 VPI specs (~12W ASIC @ 400Gbps); Optics: QSFP-DD MSA spec (10-12W @ 400G), Minkenberg et al., **IET Optoelectronics** vol. 15, 2021; Switch: Quantum-2 QM9790 datasheet (640W/51.2Tbps), Chen et al., **ISCA 2024** |

**Note on NVLink-C2C vs NVLink 4.0:** The 1.3 pJ/bit figure is specifically from the NVLink-C2C on-package interconnect (Grace Hopper). NVIDIA has not disclosed pJ/bit for NVLink 4.0 GPU-to-GPU cable SerDes. The intra-node rate (1.3 pJ/bit) serves as a lower bound; the rack-scale rate (5.0 pJ/bit) is an estimate for longer-reach copper.

#### New: `EnergyConfig` dataclass (`comm_profile.py`)

```python
@dataclass(frozen=True)
class EnergyConfig:
    nvlink_intra_node_pj_per_bit: float = 1.3
    nvlink_rack_scale_pj_per_bit: float = 5.0
    ib_pj_per_bit: float = 70.0
    rack_scale_threshold: int = 8       # GPUs/node boundary
```

Auto-loaded from `energy_config.yaml` if present in the working directory, or from `--energy-config <path>`.

#### New: `get_comm_energy()` function (`comm_profile.py`)

Returns communication energy in **microjoules (μJ)**, matching the compute energy unit from profiled CSVs (`avg_energy(uJ)`).

**Formula:** `energy = total_bytes_moved * 8 bits/byte * pJ_per_bit`, converted pJ → μJ (÷ 1e6).

**Total bytes moved** uses `_bandwidth_factor(comm_type, N)` — the same factor as the analytical comm time model:
- AllReduce: `2*(N-1)/N * data_size * N` total across all GPUs
- AllGather, ReduceScatter, AllToAll: `(N-1)/N * data_size * N`

This counts each byte **once per link traversal** (TX+RX combined), consistent with ISSCC link energy conventions.

**NVLink rate selection:** Threshold-based on `num_gpus_per_node`:
- ≤8 GPUs/node → `nvlink_intra_node_pj_per_bit` (1.3, DGX-internal)
- &gt;8 GPUs/node → `nvlink_rack_scale_pj_per_bit` (5.0, NVL72 cables)

**Multi-node (IB) decomposition:** Mirrors the hierarchical time model. For each collective, the data moved decomposes into:
- **Intra-node phase** (NVLink): `_bandwidth_factor(comm_type, num_gpus_per_node)` per GPU, at the intra-node NVLink rate
- **Inter-node phase** (IB): remaining data (full factor minus intra factor) per GPU, at the IB rate

For AllToAll(64) on 8×8 IB: 88.9% of data is intra-node NVLink, 11.1% is IB. Effective energy rate = `0.889 * 1.3 + 0.111 * 70.0 = 8.9 pJ/bit` vs `5.0 pJ/bit` for NVL64, yielding a 1.79× comm energy ratio.

#### Modified: `simulator.py` — append comm energy alongside comm time

After each `execution_time.append(comm_time)` in `get_stage_execution_time()`, a matching `execution_energy.append(comm_energy)` is added. This was the missing link — previously, execution_energy only tracked compute energy.

`Simulator.__init__` now accepts an `energy_config` parameter (threaded from `main.py` → `SearchEngine` → `Simulator`).

#### Validation

| Config | Compute Energy (KJ) | Comm Energy (KJ) | Total (KJ) | Comm % |
|---|---|---|---|---|
| DeepSeek-V3 TP1×EP64, NVL64 (64 H100, `--frequency 1980`) | 4,955 | 37 | 4,992 | 0.7% |
| DeepSeek-V3 TP1×EP64, IB 8×8 (same workload) | 4,955 | 66 | 5,022 | 1.3% |

Communication energy is a small fraction of total energy (compute dominates), but the 1.79× comm energy ratio between IB and NVL64 matches the analytical prediction. The implied NVLink power during AllToAll is ~13 W/GPU, well within the H100 NVLink power budget (~50-100W).

---

### 7. `main.py` — CLI arguments and config plumbing (~70 lines changed)

- Added `from apex_plus.simulator.comm_profile import InterconnectConfig`
- New CLI arguments:
  ```
  --interconnect {nvlink, ib}   # Default: nvlink if single-node, ib if multi-node
  --ib-rails <int>              # Default: 8 (DGX H100 = 8x ConnectX-7)
  --force-ep <int>              # Force minimum EP degree with training constraint (TP × EP = total)
  --demand-matrix <dir>         # Output directory for R×R byte-traffic matrices
  --dm-num-tokens <int>         # Representative token count (default: prompt_len)
  --energy-config <path>        # Path to YAML energy config (default: energy_config.yaml if present)
  ```
- Constructs `InterconnectConfig` and passes through both encoder and decoder `SearchEngine` instantiations
- After search, if `--demand-matrix` is set, calls `extract_demand_matrices` on the best plan and writes output files
- Added model shortcuts: `llama3.1-405b`, `moe-64x`, `moe-64x-large`, `deepseek-v2`, `deepseek-v3`

---

### 8. `apex_plus/search/engine.py` — Threading interconnect + force-EP + plan tag (~35 lines changed)

- `SearchEngine.__init__`: accepts `interconnect=None`, passes to `Simulator()`
- `search()`: Added `force_ep` parameter with training constraint (see section 4). Fixed `simulate()` return handling for bare `None` vs tuple.
- Added `get_plan_tag(plan)` helper that returns a string like `"dp1.pp1.mqa64x1.swimoe1x64"` for demand matrix file naming.

---

### 9. `apex_plus/models/registry.py` — MoE model loading fix (~20 lines changed) (pre-existing)

**Bug fixed:** The original code passed hardcoded `num_experts=1, topk=1, capacity_factor=1` to MoE model constructors instead of the user-provided values. For native MoE architectures like Mixtral, this caused a crash since `from_hf()` requires these parameters. Additionally, the dense-to-MoE conversion path attempted to pass MoE args to model classes (LLaMA, GPT, etc.) that don't accept them, causing a TypeError.

**Fix:**
- Added `_NATIVE_MOE_ARCHS` set to detect native MoE architectures (currently `MixtralForCausalLM`). These read expert config directly from the HuggingFace `config.json`.
- Added `_MOE_CONVERTIBLE_ARCHS` dict mapping architectures that support dense-to-MoE conversion (currently only `OPTForCausalLM` → `OPTMoE`). Attempting to convert unsupported architectures now raises a clear `ValueError`.

### 10. `apex_plus/models/mixtral.py` — Remove extraneous dense MLP from Mixtral IR (pre-existing)

**Bug fixed:** `to_ir()` included both a dense `MLP` cell and a `SwiMoE` cell in each decoder block. In standard Mixtral, the MoE FFN **replaces** the dense MLP — there is no separate dense MLP per block. Including both caused the simulator to double-count FFN computation for every Mixtral layer.

**Fix:** Removed the `MLP` cell from the decoder block. Block now contains `[MQA, SwiMoE]` only. Removed unused `MLP` import.

---

### 11. New model config files (added for evaluation)

| File | Description |
|---|---|
| `apex_plus/models/llama3.1_405b_config.json` | LLaMA 3.1 405B (128 heads, 8 KV heads, 126 layers, hidden=16384, intermediate=53248) |
| `apex_plus/models/moe_64x_config.json` | 64-expert MoE, small experts (intermediate=2048, hidden=7168, top-6, 60 layers) |
| `apex_plus/models/moe_64x_large_config.json` | 64-expert MoE, large experts (intermediate=16384, hidden=7168, top-6, 60 layers) |
| `apex_plus/models/deepseek_v2_config.json` | DeepSeek-V2 (160 experts, intermediate=1536, top-6, hidden=5120, 80 heads, 60 layers) |
| `apex_plus/models/deepseek_v3_config.json` | DeepSeek-V3 (256 experts, intermediate=2048, top-8, hidden=7168, 112 heads, 61 layers) |

---

## Known Limitations and Modeling Assumptions

### Analytical NVLink model
- **Bandwidth extraction** from 8-GPU profiled data yields ~317 GB/s for AllToAll, vs theoretical ~450 GB/s NVLink injection rate. This under-represents NVSwitch performance for AllToAll specifically, because the 8-GPU profile captures NCCL ring/mesh overhead that doesn't apply to NVSwitch crossbar at scale.
- The `(N-1)/N` bandwidth factor is correct (information-theoretic data volume), but the extracted per-GPU bandwidth from small-scale profiles may be pessimistic for large-scale NVSwitch.

### IB hierarchical model
- Assumes uniform IB bandwidth between all node pairs (no fat-tree congestion, no oversubscription).
- Multi-rail scheduling assumes ideal 1:1 GPU-to-NIC mapping.
- PCIe traversal latency (GPU <-> NIC, ~2-3 us) is partially absorbed into the `ib_latency_us` parameter but not explicitly modeled.
- No modeling of network congestion from concurrent collectives.

### MoE / Expert Parallelism
- **AllToAll assumes uniform token distribution** across experts. In practice, router decisions create skewed distributions where popular experts receive disproportionately more tokens. The `capacity_factor` parameter scales data volume but does not model traffic asymmetry.
- AllToAll completion time is bounded by the slowest GPU pair in reality, but modeled as a symmetric collective here.
- This uniformity assumption particularly under-estimates the NVL64 advantage over IB: NVSwitch's non-blocking crossbar handles non-uniform AllToAll traffic gracefully, while IB fat-trees suffer hotspots under skewed patterns.
- EP is a binary choice per cell (DefaultTemplate vs MoETemplate0), not an orthogonal parallelism dimension combinable with TP.

### GEMM profile extrapolation
- Linear extrapolation beyond `MAX_NUM_INPUT_TOKENS = 65536` assumes compute-bound regime. For very small m or k dimensions with large n, the GEMM may be memory-bandwidth-bound, where linear extrapolation over-estimates time. In practice, the MoE expert GEMMs (m=intermediate_size, k=hidden_size) are compute-bound at the relevant n values.

### Training constraint (`--force-ep`)
- The constraint `attention cell-DP = MoE EP` correctly enforces `TP × EP = total` for the common case. However, it does not model overlapping TP and EP groups (e.g., TP4 × EP16 on 64 GPUs where TP groups are subsets of EP groups). In APEX+'s cell-level model, such configurations would still require resharding.
- The `--force-ep` flag only constrains the parallelism search; it does not change how tokens are routed or how communication volumes are computed.

### Communication energy model
- **NVLink rate selection is threshold-based** (≤8 GPUs/node = intra-node, >8 = rack-scale). This is a coarse proxy for physical link length. Systems with 8 GPUs on a single board but using copper cables between GPU trays (e.g., some NVL36 configurations) may warrant a different threshold.
- **The 1.3 pJ/bit intra-node rate** is from NVLink-C2C (on-package, Grace Hopper), which uses different signaling than NVLink 4.0 GPU-to-GPU within a DGX. The actual NVLink 4.0 intra-node energy is likely higher (2-3 pJ/bit) but NVIDIA has not disclosed it.
- **The 5.0 pJ/bit rack-scale rate** is an estimate, not a measured value. Actual NVLink 4.0 copper cable SerDes energy is unknown.
- **IB energy uses a single aggregate rate** (70 pJ/bit) for the full path (NIC + optics + switch). This does not account for: (a) varying hop counts in fat-tree topologies, (b) passive vs active cables, (c) congestion-induced retransmissions. The individual components (NIC ~30, optics ~25, switch ~12 pJ/bit) are derived from datasheets, not direct measurements.
- **Energy is per-link (TX+RX combined)**, matching ISSCC reporting conventions. Each byte moving from GPU A to GPU B is counted once. The `_bandwidth_factor` accounts for protocol overhead (AllReduce counts 2x due to reduce + broadcast).
- **No overlap modeling**: Communication and compute energy are added independently. In practice, NVLink and GPU compute share power delivery, and concurrent comm+compute may have non-linear power interactions.

### Demand matrix extraction
- Matrices represent **per-block** aggregate traffic (dispatch + combine AllToAll summed together). Each reshard chain executes once per block, so `total_per_iteration = matrix_entry * num_blocks_per_stage`.
- Zipf load ratios assume round-robin expert-to-GPU assignment sorted by popularity. Real systems may use different assignment strategies.
- DP replicas produce zero cross-replica traffic (inference only). Training gradient sync is not modeled.
- Device group assignment assumes contiguous partitioning (matching `Cluster.partition()`). Non-contiguous rank layouts would require a different group mapping.

### General
- Existing profiled configurations (1-8 GPUs, single-node) are completely unaffected — they still use the CSV lookup path.
- The analytical models are most accurate for NVSwitch-based systems. Multi-node estimates should be validated against actual profiling data when available.

---

## Validation Results

### DeepSeek-V3 on 64 H100 GPUs, TP1 × EP64 (training-style, `--force-ep 64`)
*Workload: prompt=8192, output=1, 4096 requests (prefill-dominated)*

| Metric | NVL64 | 8×8 IB | Delta |
|---|---|---|---|
| Throughput (tok processed/s) | 345,271 | 303,702 | **NVL64 +13.7%** |
| MQA time (s) | 50.85 | 50.85 | Same (TP1, no comm) |
| AllToAll time (s) | 22.78 | 37.98 | **NVL64 40% faster** |
| SwiMoE compute (s) | 37.45 | 37.45 | Same |
| AllToAll % of total | 23.4% | 34.4% | — |
| Total time (s) | 97.19 | 110.50 | **NVL64 12.0% faster** |
| Optimal plan | TP1 × EP64 | same | — |

With TP1, attention has **zero communication** — MQA time is identical on both platforms. AllToAll is 23-34% of total time, approaching the ~55% reported in training literature. The remaining gap comes from: (a) uniform token distribution assumption (real routing is skewed, increasing AllToAll time), (b) output_len=1 (no decode phase where AllToAll dominates more), and (c) no backward pass modeling.

### LLaMA 3.1 405B on 64 H100 GPUs (prompt=2048, output=128, 64 requests)

| Metric | NVL64 | 8×8 IB | Delta |
|---|---|---|---|
| Throughput (tok gen/s) | 859 | 825 | NVL64 +4.0% |
| AllGather time (s) | 1.21 | 1.56 | NVL64 29% faster |
| Total time (s) | 9.54 | 9.93 | NVL64 4% faster |
| Optimal plan | TP8 x cell-DP2 x PP2 x DP2 | same | — |

Cross-node AllGather (TP16 for GLU cell spanning 2 nodes on IB) shows the expected IB penalty.

### 64-Expert MoE (large experts) on 64 H100 GPUs (prefill-heavy: prompt=8192, output=1, 4096 requests)

| Metric | NVL64 | 8×8 IB | Delta |
|---|---|---|---|
| Throughput (tok processed/s) | 111,268 | 108,571 | NVL64 +2.5% |
| AllToAll time (s) | 16.46 | 22.98 | NVL64 28% faster |
| SendRecv/PP time (s) | 0.17 | 1.20 | NVL64 86% faster |
| Total time (s) | 301.6 | 309.1 | NVL64 2.4% faster |
| Optimal plan | PP4 x TP2 x EP16 | same | — |

AllToAll shows significant NVLink advantage in absolute terms, but overall gap is modest because SwiMoE compute dominates (73% of total time). This uses the optimizer-chosen plan (not `--force-ep`), so EP is only 16, not 64.

### Regression: 8-GPU single-node LLaMA3-70B

Results are identical to before changes — profiled CSV path is unaffected.

---

## Usage Examples

```bash
# NVL64 (single-node, 64 GPUs, NVLink — default interconnect)
python main.py --model llama3.1-405b --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 2048 --output-len 128

# 8-node x 8-GPU IB cluster (auto-selects IB when num_nodes > 1)
python main.py --model llama3.1-405b --num-nodes 8 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --prompt-len 2048 --output-len 128

# Explicit IB with custom rail count
python main.py --model moe-64x-large --num-nodes 8 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --interconnect ib --ib-rails 4 \
  --prompt-len 8192 --output-len 1 --num-requests 4096

# Force NVLink model even for multi-node (for NVL-style multi-node setups)
python main.py --model llama3-70b --num-nodes 2 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --interconnect nvlink

# Training-style: DeepSeek-V3 with forced TP1 x EP64
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64

# Training-style comparison: same model on 8x8 IB
python main.py --model deepseek-v3 --num-nodes 8 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64

# Extract demand matrices (R×R byte-traffic per collective)
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --demand-matrix ../profiling/demand_matrix --dm-num-tokens 8192

# Demand matrices with Zipf-skewed MoE routing
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --demand-matrix ../profiling/demand_matrix --dm-num-tokens 8192 --moe-skew 1.0

# Energy modeling (requires --frequency for compute energy; comm energy always active)
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --frequency 1980

# Custom energy config
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --frequency 1980 --energy-config energy_config.yaml

# Comm-only energy (--frequency 0 disables compute energy)
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --frequency 0
```

---

## 9. Pluggable Expert Routing Distribution

### Problem

AllToAll demand matrix rows were identical: every source GPU applied the same
`load_ratios[j]` scaling to the same `total_volume / n`, producing uniform
rows even under Zipf skew. Real MoE routing has per-GPU variation because
each GPU independently routes its tokens to experts.

### Solution

Extracted AllToAll traffic modeling into a pluggable `ExpertDistribution`
interface (`apex_plus/simulator/expert_distribution.py`).

**Interface:**
```python
class ExpertDistribution(ABC):
    def route_tokens(self, src_idx: int, num_tokens: int, group_size: int) -> np.ndarray:
        """Return token counts routed from src GPU to each of group_size destinations."""
```

**Built-in implementations:**
- `UniformDistribution` — each destination equally likely; multinomial sampling adds noise
- `ZipfDistribution` — experts ranked by Zipf(s), assigned round-robin to GPUs; multinomial sampling per source GPU
- `DirichletDistribution` — each source GPU draws its own probability vector from Dirichlet(α); models learned routers with heterogeneous per-GPU gating

Each source GPU independently samples `Multinomial(num_tokens, gpu_probs)`,
so rows naturally differ (realistic token-level routing variance).

A `DISTRIBUTION_REGISTRY` maps CLI names to classes. New distributions only
need to subclass `ExpertDistribution` and register themselves.

### CLI arguments

```
--moe-dist {uniform,zipf,dirichlet}   Distribution name
--moe-dist-param FLOAT                Distribution-specific parameter:
                                         zipf: exponent s (0=uniform, 1.0=classic Zipf)
                                         dirichlet: concentration alpha (<1 spiky, >10 flat)
                                         uniform: ignored
--moe-skew FLOAT                       (Legacy) equivalent to --moe-dist zipf --moe-dist-param <value>
```

### Files changed

| File | Change |
|---|---|
| `apex_plus/simulator/expert_distribution.py` | NEW — `ExpertDistribution` ABC + `UniformDistribution` + `ZipfDistribution` + `DirichletDistribution` + `DISTRIBUTION_REGISTRY` + `make_expert_dist()` factory |
| `apex_plus/simulator/demand_matrix.py` | Removed `_zipf_gpu_probs`, rewrote `_fill_alltoall` to use `ExpertDistribution`; added `expert_dist` kwarg to `extract_demand_matrices()` |
| `main.py` | Added `--moe-dist` and `--moe-dist-param` CLI arguments; wires factory through to demand matrix extraction |

### Usage

```bash
# CLI: Zipf distribution
python main.py --model qwen3-235b ... --moe-dist zipf --moe-dist-param 1.0 --demand-matrix ../profiling/demand_matrix

# CLI: Dirichlet distribution (spiky per-GPU routing)
python main.py --model qwen3-235b ... --moe-dist dirichlet --moe-dist-param 0.5 --demand-matrix ../profiling/demand_matrix

# CLI: Legacy syntax (backward-compatible)
python main.py --model qwen3-235b ... --moe-skew 1.0 --demand-matrix ../profiling/demand_matrix
```

```python
# Programmatically with factory:
from apex_plus.simulator.expert_distribution import make_expert_dist
dist = make_expert_dist("dirichlet", param=0.5)
matrices = extract_demand_matrices(plan, cluster, model, dtype, tokens, expert_dist=dist)

# Or directly:
from apex_plus.simulator.expert_distribution import DirichletDistribution
dist = DirichletDistribution(alpha=0.3, seed=42)
matrices = extract_demand_matrices(plan, cluster, model, dtype, tokens, expert_dist=dist)
```
