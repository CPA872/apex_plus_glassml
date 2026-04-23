import math
import warnings
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from apex_plus.parallel.comm import CommType
from apex_plus.utils.dtype import DTYPE, dtype_to_str

KB = 1024


@dataclass(frozen=True)
class InterconnectConfig:
    mode: str = "nvlink"            # "nvlink" or "ib"
    ib_bw_per_rail_GBs: float = 50.0   # 400Gbps = 50 GB/s per rail
    num_rails: int = 8              # DGX H100 has 8 ConnectX-7 ports
    ib_latency_us: float = 3.0     # base IB RDMA latency per message

    @property
    def ib_bw_per_rail_bytes_per_us(self) -> float:
        return self.ib_bw_per_rail_GBs * 1e9 / 1e6  # bytes/us


@dataclass(frozen=True)
class EnergyConfig:
    nvlink_intra_node_pj_per_bit: float = 1.3   # <=8 GPUs/node (DGX-internal)
    nvlink_rack_scale_pj_per_bit: float = 5.0   # >8 GPUs/node (NVL72 cables)
    ib_pj_per_bit: float = 70.0                  # Full IB path (NIC + optics + switch)
    rack_scale_threshold: int = 8                 # GPUs/node boundary

_COMM_TYPE_TO_OP_KIND = {
    CommType.AllGather: "allgather",
    CommType.AllReduce: "allreduce",
    CommType.AllToAll: "alltoall",
    CommType.ReduceScatter: "reducescatter",
}

ALL_GATHER_TMPL = "profile/comm/{gpu}/all_gather.csv"
ALL_REDUCE_TMPL = "profile/comm/{gpu}/all_reduce.csv"
ALL_TO_ALL_TMPL = "profile/comm/{gpu}/alltoall.csv"
REDUCE_SCATTER_TMPL = "profile/comm/{gpu}/reduce_scatter.csv"
SEND_RECV_TMPL = "profile/comm/{gpu}/sendrecv.csv"


@lru_cache(maxsize=512)
def _load_table(op_kind: str, gpu: str) -> pd.DataFrame:
    """
    op_kind ∈ {"allgather", "allreduce", "alltoall", "reducescatter", "sendrecv}
    """
    name = {"allgather": ALL_GATHER_TMPL, "allreduce": ALL_REDUCE_TMPL, "alltoall": ALL_TO_ALL_TMPL, "reducescatter": REDUCE_SCATTER_TMPL, "sendrecv": SEND_RECV_TMPL}[op_kind].format(
        gpu = gpu
    )
    df = pd.read_csv(name)

    return df


def _allgather_df(gpu: str) -> pd.DataFrame:
    return _load_table("allgather", gpu)

def _allreduce_df(gpu: str) -> pd.DataFrame:
    return _load_table("allreduce", gpu)

def _alltoall_df(gpu: str) -> pd.DataFrame:
    return _load_table("alltoall", gpu)

def _reducescatter_df(gpu: str) -> pd.DataFrame:
    return _load_table("reducescatter", gpu)

def _sendrecv_df(gpu: str) -> pd.DataFrame:
    return _load_table("sendrecv", gpu)


def _bandwidth_factor(comm_type: CommType, n: int) -> float:
    """Return the multiplicative factor on (data_size / bandwidth) for each collective."""
    if comm_type == CommType.AllReduce:
        return 2.0 * (n - 1) / n
    elif comm_type in (CommType.AllGather, CommType.ReduceScatter, CommType.AllToAll):
        return (n - 1) / n
    else:
        raise ValueError(f"Unknown comm type for analytical model: {comm_type}")


@lru_cache(maxsize=256)
def _extract_bandwidth_and_latency(
    comm_type: CommType,
    op_kind: str,
    gpu: str,
    dtype_str: str,
    ref_n: int = 8,
) -> tuple:
    """Extract per-GPU algo bandwidth and latency from reference N-GPU profiled data."""
    df = _load_table(op_kind, gpu)

    # Try ref_n, then fall back to smaller counts
    for n in [ref_n, 4, 2]:
        ref_df = df[
            (df["num_nodes"] == 1)
            & (df["num_gpus_per_node"] == n)
            & (df["dtype"] == dtype_str)
        ]
        if not ref_df.empty:
            ref_n = n
            break
    else:
        raise ValueError(
            f"No reference data found for analytical model: "
            f"op={op_kind}, gpu={gpu}, dtype={dtype_str}"
        )

    ref_df = ref_df.sort_values("size(kb)")
    large1 = ref_df.iloc[-1]  # largest message
    large2 = ref_df.iloc[-2]  # second largest

    factor = _bandwidth_factor(comm_type, ref_n)
    delta_size_bytes = (large1["size(kb)"] - large2["size(kb)"]) * 1024
    delta_time = large1["time(us)"] - large2["time(us)"]

    if delta_time <= 0 or delta_size_bytes <= 0:
        # Degenerate case: use single-point estimate
        algo_bw = factor * large1["size(kb)"] * 1024 / max(large1["time(us)"], 1.0)
    else:
        algo_bw = factor * delta_size_bytes / delta_time  # bytes/us

    # Back-compute latency from largest data point
    latency = large1["time(us)"] - factor * large1["size(kb)"] * 1024 / algo_bw
    latency = max(latency, ref_df.iloc[0]["time(us)"])  # floor at zero-size time

    return algo_bw, latency, ref_n


def _analytical_comm_time(
    comm_type: CommType,
    gpu: str,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype_str: str,
    size_kb: int,
) -> float:
    """Estimate comm time for unprofiled GPU count using analytical model."""
    if num_nodes > 1:
        warnings.warn(
            f"Analytical comm model assumes NVLink domain but num_nodes={num_nodes}. "
            f"Cross-node (IB/EFA) estimates may be inaccurate.",
            stacklevel=3,
        )

    target_n = num_nodes * num_gpus_per_node
    op_kind = _COMM_TYPE_TO_OP_KIND[comm_type]
    algo_bw, ref_latency, ref_n = _extract_bandwidth_and_latency(
        comm_type, op_kind, gpu, dtype_str
    )

    # Single-node NVSwitch is a non-blocking crossbar: latency is constant
    # regardless of GPU count (single hop). Multi-node uses tree/ring
    # algorithms where latency scales logarithmically.
    if num_nodes == 1:
        latency = ref_latency
    else:
        latency = ref_latency * math.log2(max(target_n, 2)) / math.log2(max(ref_n, 2))

    # Compute bandwidth term with target N
    target_factor = _bandwidth_factor(comm_type, target_n)
    bw_term = target_factor * size_kb * 1024 / algo_bw if algo_bw > 0 else 0.0

    return max(latency + bw_term, latency)


def _ib_phase_time(
    comm_type: CommType,
    num_nodes: int,
    data_bytes_per_rail: float,
    config: InterconnectConfig,
) -> float:
    """Compute time for the inter-node IB phase of a hierarchical collective."""
    bw = config.ib_bw_per_rail_bytes_per_us
    # Modern NCCL uses parallel non-blocking RDMA writes for all collectives.
    # AllToAll: NIC handles concurrent QPs to all peers simultaneously,
    # so latency depth is O(log N), not O(N) serial round-trips.
    # Other collectives (AllReduce, AllGather, ReduceScatter) use tree/ring.
    latency = config.ib_latency_us * math.ceil(math.log2(max(num_nodes, 2)))
    factor = _bandwidth_factor(comm_type, num_nodes)
    bw_term = factor * data_bytes_per_rail / bw if bw > 0 else 0.0
    return latency + bw_term


def _hierarchical_comm_time(
    comm_type: CommType,
    gpu: str,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype: DTYPE,
    num_elements: int,
    config: InterconnectConfig,
) -> float:
    """Model multi-node collective as hierarchical: NVLink phases + IB phase."""
    total_data_bytes = num_elements * dtype.size
    data_per_rail = total_data_bytes / config.num_rails

    # Intra-node phases reuse profiled/analytical NVLink model (num_nodes=1)
    def nvlink_time(ct, n_elems):
        return get_comm_time(ct, gpu, 1, num_gpus_per_node, dtype, n_elems)

    if comm_type == CommType.AllReduce:
        # Phase 1: Intra-node ReduceScatter on full E elements (NVLink)
        t1 = nvlink_time(CommType.ReduceScatter, num_elements)
        # Phase 2: Inter-node AllReduce (IB, k parallel rings each handling E/k data)
        # After RS each GPU has E/k; with k=num_rails rings, data_per_rail = E/k
        t2 = _ib_phase_time(CommType.AllReduce, num_nodes, data_per_rail, config)
        # Phase 3: Intra-node AllGather back to full E elements (NVLink)
        t3 = nvlink_time(CommType.AllGather, num_elements)
        return t1 + t2 + t3

    elif comm_type == CommType.AllGather:
        # Hierarchical AG: inter-node first (small), then intra-node (large).
        # Phase 1: Intra-node AllGather on full E elements (NVLink)
        # After inter-node AG delivers E/k per GPU, local AG reconstructs full E.
        # Per-GPU NVLink traffic = (k-1)/k * E.
        t1 = nvlink_time(CommType.AllGather, num_elements)
        # Phase 2: Inter-node AllGather (IB, k parallel rings each handling E/k)
        # Each rank gathers its chunk from peer ranks on other nodes.
        # Per-rail traffic = (m-1)/m * E/k.
        t2 = _ib_phase_time(CommType.AllGather, num_nodes, data_per_rail, config)
        return t1 + t2

    elif comm_type == CommType.ReduceScatter:
        # Hierarchical RS: intra-node first (large), then inter-node (small).
        # Phase 1: Inter-node ReduceScatter (IB, k parallel rings each handling E/k)
        # Per-rail traffic = (m-1)/m * E/k.
        t1 = _ib_phase_time(CommType.ReduceScatter, num_nodes, data_per_rail, config)
        # Phase 2: Intra-node ReduceScatter on full E elements (NVLink)
        # RS across k GPUs reduces full E → E/k per GPU.
        # Per-GPU NVLink traffic = (k-1)/k * E.
        t2 = nvlink_time(CommType.ReduceScatter, num_elements)
        return t1 + t2

    elif comm_type == CommType.AllToAll:
        # Phase 1: Intra-node AllToAll on E/m elements (NVLink)
        # Each GPU sends E/N to each of k-1 local peers
        t1 = nvlink_time(CommType.AllToAll, num_elements // num_nodes)
        # Phase 2: Inter-node AllToAll (IB)
        # AllToAll is point-to-point, not ring-based: each GPU's NIC carries
        # ALL of that GPU's inter-node traffic. No multi-rail splitting like
        # ring collectives. Per-NIC load = (m-1)/m * E, not E/num_rails.
        t2 = _ib_phase_time(CommType.AllToAll, num_nodes, total_data_bytes, config)
        return t1 + t2

    else:
        raise ValueError(f"Unsupported comm type for hierarchical model: {comm_type}")


def _interpolate(
    df: pd.DataFrame,
    col: str,
    val: int,
    target_col: str,
) -> float:
    large = df[df[col] >= val]
    if large.empty:
        r = val / df[col].max()
        return df[target_col].max() * r

    small = df[df[col] <= val]
    if len(small) == 0:
        raise ValueError(f"Cannot interpolate. {col}={val}")

    small = small.iloc[-1]
    large = large.iloc[0]
    if small[col] == large[col]:
        return small[target_col]

    r = (val - small[col]) / (large[col] - small[col])
    return small[target_col] * (1 - r) + large[target_col] * r


@lru_cache(maxsize=512)
def get_comm_time(
    comm_type: CommType,
    gpu: str,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype: DTYPE,
    num_elements: int,
    interconnect: InterconnectConfig = None,
) -> float:
    if num_nodes * num_gpus_per_node <= 1:
        return 0.0

    # Multi-node IB: use hierarchical decomposition
    if num_nodes > 1 and interconnect is not None and interconnect.mode == "ib":
        return _hierarchical_comm_time(
            comm_type, gpu, num_nodes, num_gpus_per_node,
            dtype, num_elements, interconnect,
        )

    if comm_type == CommType.AllGather:
        df = _allgather_df(gpu)
    elif comm_type == CommType.AllReduce:
        df = _allreduce_df(gpu)
    elif comm_type == CommType.AllToAll:
        df = _alltoall_df(gpu)
    elif comm_type == CommType.ReduceScatter:
        df = _reducescatter_df(gpu)
    else:
        raise ValueError(f"Unknown comm type: {comm_type}")

    dtype_str = "half" if dtype == DTYPE.FLOAT8 else dtype_to_str(dtype)
    size = num_elements * dtype.size // KB

    df_filtered = df[df["num_nodes"] == num_nodes]
    df_filtered = df_filtered[df_filtered["num_gpus_per_node"] == num_gpus_per_node]
    df_filtered = df_filtered[df_filtered["dtype"] == dtype_str]

    if not df_filtered.empty:
        return _interpolate(df_filtered, "size(kb)", size, "time(us)")

    # Analytical fallback for unprofiled GPU counts (NVLink domain)
    return _analytical_comm_time(
        comm_type, gpu, num_nodes, num_gpus_per_node, dtype_str, size
    )


def get_comm_energy(
    comm_type: CommType,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype: DTYPE,
    num_elements: int,
    energy_config: EnergyConfig = None,
) -> float:
    """Return communication energy in microjoules (uJ) for one comm group.

    Energy = total_bytes_moved_across_all_GPUs * 8 bits/byte * pJ_per_bit,
    converted from pJ to uJ (÷ 1e6).

    Matches the unit used by compute energy profiles (avg_energy(uJ) in CSVs).

    For multi-node (IB) collectives, the intra-node NVLink phases use the
    NVLink rate and the inter-node IB phase uses the IB rate.
    """
    if energy_config is None or num_nodes * num_gpus_per_node <= 1:
        return 0.0

    total_data_bytes = num_elements * dtype.size
    n = num_nodes * num_gpus_per_node
    factor = _bandwidth_factor(comm_type, n)
    moved_bytes_per_gpu = factor * total_data_bytes
    total_moved_bytes = moved_bytes_per_gpu * n

    if num_nodes == 1:
        # Pure NVLink — choose rate based on GPUs/node threshold
        if num_gpus_per_node > energy_config.rack_scale_threshold:
            pj_per_bit = energy_config.nvlink_rack_scale_pj_per_bit
        else:
            pj_per_bit = energy_config.nvlink_intra_node_pj_per_bit
        pj = total_moved_bytes * 8 * pj_per_bit
    else:
        # Hierarchical: intra-node NVLink + inter-node IB
        # Ring collectives (AR/AG/RS): NVLink phase operates on full tensor E,
        # so intra = bw_factor(type, k) * E. The identity (total - intra) then
        # yields the small IB fraction = (m-1)/(mk) * E.
        # AllToAll (point-to-point): NVLink phase operates on E/m,
        # so intra = bw_factor(A2A, k) * E/m. IB fraction = (m-1)/m * E.
        if num_gpus_per_node > 1:
            intra_factor = _bandwidth_factor(comm_type, num_gpus_per_node)
            if comm_type == CommType.AllToAll:
                intra_bytes_per_gpu = intra_factor * total_data_bytes / num_nodes
            else:
                intra_bytes_per_gpu = intra_factor * total_data_bytes
        else:
            intra_bytes_per_gpu = 0.0
        inter_bytes_per_gpu = moved_bytes_per_gpu - intra_bytes_per_gpu

        # Intra-node NVLink uses intra-node rate (<=8 GPUs within DGX)
        nvlink_pj = intra_bytes_per_gpu * n * 8 * energy_config.nvlink_intra_node_pj_per_bit
        ib_pj = inter_bytes_per_gpu * n * 8 * energy_config.ib_pj_per_bit
        pj = nvlink_pj + ib_pj

    return pj / 1e6  # pJ → uJ


@lru_cache(maxsize=256)
def get_p2p_comm_time(
    gpu: str,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype: DTYPE,
    num_elements: int,
    interconnect: InterconnectConfig = None,
) -> float:
    if num_nodes == 1 and num_gpus_per_node == 1:
        return 0.0

    # Cross-node P2P over IB: analytical model
    if num_nodes >= 2 and interconnect is not None and interconnect.mode == "ib":
        data_bytes = num_elements * dtype.size
        return interconnect.ib_latency_us + data_bytes / interconnect.ib_bw_per_rail_bytes_per_us

    if num_nodes * num_gpus_per_node != 2:
        raise ValueError("P2P communication only supports 2 GPUs.")

    df = _sendrecv_df(gpu)

    df = df[df["num_nodes"] == num_nodes]
    df = df[df["num_gpus_per_node"] == num_gpus_per_node]
    dtype_str = "half" if dtype == DTYPE.FLOAT8 else dtype_to_str(dtype)
    df = df[df["dtype"] == dtype_str]
    assert not df.empty, (
        f"Cannot find Send Recv comm time for "
        f"gpu={gpu}, num_nodes={num_nodes}, "
        f"num_gpus_per_node={num_gpus_per_node}, dtype={dtype_str}"
    )

    size = num_elements * dtype.size // KB
    return _interpolate(df, "size(kb)", size, "time(us)")
