# NVL64 vs IB: When Does NVLink Matter?

We extended APEX+ to compare 64 H100 GPUs in a single NVLink domain (NVL64) against 8 nodes x 8 GPUs connected by 400Gbps InfiniBand. The goal: quantify when NVL64's higher interconnect bandwidth actually translates to real throughput gains.

## TL;DR

NVL64 helps when **AllToAll communication is a large fraction of total time**. This happens with fine-grained MoE models at high expert parallelism. For dense models or fat-expert MoE, compute dominates and the interconnect barely matters.

## Results

| Model | Config | NVL64 vs IB | Why |
|---|---|---|---|
| LLaMA 3.1 405B | TP8, dense | **+4%** | Compute-dominated, AllGather is ~12% of time |
| 64-Expert MoE (large) | Optimizer-chosen EP16 | **+2.4%** | Fat experts, compute is 73% of time |
| DeepSeek-V3 (256 experts) | Forced TP1 x EP64 | **+126%** | Fine-grained experts, AllToAll is 28% (NVL64) vs 72% (IB) of time |

AllToAll is **~5.7× faster on NVL64** vs 8-rail IB. This reflects the per-GPU bandwidth gap: NVLink ~317 GB/s vs single-NIC IB 50 GB/s. AllToAll is point-to-point (no multi-rail ring splitting), so each GPU's NIC is the bottleneck. The overall speedup depends on how much of total time is spent in AllToAll.

## What Drives NVL64 Advantage

| Factor | More NVL64 benefit | Less NVL64 benefit |
|---|---|---|
| Expert size | Small (DeepSeek-V3: 2048 intermediate) | Large (16384 intermediate) |
| Expert count | Many (256) | Few (64) |
| EP degree | High (EP64) | Low (EP8-16, optimizer prefers DP) |
| AllToAll fraction | High (>25% of time) | Low (<10% of time) |
| Workload | Prefill-heavy, large batches | Decode-heavy, small messages |

## Demand Matrix Extraction

APEX+ can now extract **R x R byte-traffic matrices** from execution plans, bridging the gap to cycle-accurate network simulators. Each matrix entry `[i][j]` = bytes from GPU rank i to rank j per collective call.

```bash
# Extract demand matrices for DeepSeek-V3 TP1 x EP64
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --demand-matrix ../profiling/demand_matrix --dm-num-tokens 8192

# With Zipf-skewed expert routing (non-uniform AllToAll traffic)
python main.py --model deepseek-v3 ... --moe-dist zipf --moe-dist-param 1.0 --demand-matrix ../profiling/demand_matrix

# With Dirichlet routing (per-GPU heterogeneous gating)
python main.py --model deepseek-v3 ... --moe-dist dirichlet --moe-dist-param 0.5 --demand-matrix ../profiling/demand_matrix

# Legacy syntax (equivalent to --moe-dist zipf --moe-dist-param 1.0)
python main.py --model deepseek-v3 ... --moe-skew 1.0 --demand-matrix ../profiling/demand_matrix
```

Output: one text file per collective type (e.g., `ep_alltoall.txt`, `tp_allreduce.txt`), compatible with the `demand-matrix/` project format.

**Supported parallelism types:**
- **TP**: AllReduce within tensor-parallel groups
- **EP**: AllToAll with pluggable expert routing distribution (see below)
- **DP**: Block-diagonal structure (zero cross-replica traffic in inference)
- **Resharding**: ReduceScatter/AllGather between cells with different parallelism

### Pluggable Expert Routing Distributions

AllToAll traffic patterns are determined by a pluggable `ExpertDistribution` interface (`apex_plus/simulator/expert_distribution.py`). Each source GPU independently samples how many tokens it sends to each destination, creating realistic per-GPU variation (rows are NOT identical).

**CLI usage:** `--moe-dist <name> --moe-dist-param <value>`

**Built-in distributions:**

| Distribution | `--moe-dist` | `--moe-dist-param` | Traffic Pattern |
|---|---|---|---|
| Uniform | `uniform` | (ignored) | Homogeneous with multinomial noise |
| Zipf | `zipf` | exponent `s` (0.3-0.5 typical, 1.0 = classic Zipf) | Column gradient: popular-expert GPUs receive more traffic |
| Dirichlet | `dirichlet` | concentration `alpha` (<1 spiky, 1 = uniform simplex, >10 flat) | Each source GPU has its own hot-spot destinations |

**Key differences:**
- **Zipf**: All sources share the same global popularity vector; variation comes from multinomial sampling. Models well-known expert popularity skew.
- **Dirichlet**: Each source draws its *own* probability vector from Dirichlet(α). Models learned routers with heterogeneous per-GPU gating preferences.

**Adding a custom distribution:** Subclass `ExpertDistribution`, implement `route_tokens(src_idx, num_tokens, group_size) -> np.ndarray`, and register it in `DISTRIBUTION_REGISTRY`.

## Communication Energy Model

APEX+ now models communication energy alongside compute energy using pJ/bit metrics.

| Link Type | Rate | Physical Context |
|---|---|---|
| Intra-node NVLink (<=8 GPUs) | 1.3 pJ/bit | DGX-internal PCB traces. Source: NVLink-C2C, ISSCC 2023 |
| Rack-scale NVLink (>8 GPUs) | 5.0 pJ/bit | NVL72 copper cables. Estimated from PAM4 SerDes class |
| IB full path | 70.0 pJ/bit | NIC (~30) + optics (~25) + switch (~12). Derived from CX-7/QM9790 datasheets |

Configure via `config.yaml` (auto-loaded) or `--config <path>`. Use `--frequency 1980` for compute+comm energy, `--frequency 0` for comm-only.

**Key finding:** Communication energy is <1% of total for NVL64 but ~8.5% for IB, reflecting AllToAll's 5.7× latency penalty on IB. Comm-only energy ratio is 12.5× (IB vs NVL64), driven by 70 pJ/bit IB carrying 87.5% of AllToAll traffic. Total energy difference is ~8.5% (5,417 vs 4,992 KJ) because GPU compute still dominates.

## What We Changed in APEX+

1. **Analytical NVLink model** for >8 GPUs (NVSwitch = constant latency, not log-scaled)
2. **Hierarchical IB model** decomposing collectives into NVLink + IB phases
3. **`--force-ep`** flag enforcing training-style TP x EP = total GPUs
4. **GEMM extrapolation** for token counts beyond profiled range
5. **Demand matrix extraction** (`--demand-matrix`) for per-rank-pair byte traffic
6. **Communication energy modeling** with pJ/bit metrics for NVLink (intra-node/rack-scale) and IB
7. **Hierarchical IB model fixes**: AllToAll IB phase now correctly uses single-NIC bandwidth (no multi-rail splitting); ring collectives (AR/AG/RS) unchanged
8. **Pluggable expert routing** (`expert_distribution.py`): AllToAll traffic modeled via per-GPU multinomial sampling with swappable distribution backends (Zipf, uniform, or custom)
9. **Bug fixes**: Mixtral double-counting MLP, MoE registry crashes, batch size check for mixed cell-DP, demand matrix identical rows

See [CHANGES.md](CHANGES.md) for full technical details.

## Quick Start

```bash
# NVL64 (with energy: add --frequency 1980)
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --frequency 1980

# 8x8 IB (same model, same parallelism)
python main.py --model deepseek-v3 --num-nodes 8 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --frequency 1980

# Comm energy only (isolate communication cost)
python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --frequency 0
```
