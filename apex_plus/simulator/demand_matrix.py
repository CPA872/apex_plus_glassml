"""Extract R×R demand matrices from APEX+ execution plans.

Each matrix entry [i][j] = bytes sent from rank i to rank j per collective call.
Output format matches the demand-matrix/ project convention.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from apex_plus.cluster.cluster import Cluster
from apex_plus.execution.plan import ExecutionPlan
from apex_plus.ir.transformer import Transformer
from apex_plus.parallel.comm import CommType
from apex_plus.simulator.expert_distribution import (
    ExpertDistribution,
    make_expert_dist,
)
from apex_plus.utils.dtype import DTYPE


def get_device_ids(cluster: Cluster) -> List[int]:
    """Recursively extract device_id values from a Cluster."""
    if cluster.children:
        ids = []
        for child in cluster.children:
            ids.extend(get_device_ids(child))
        return ids
    return [d.device_id for d in cluster.devices]


def _make_expert_dist_legacy(
    num_experts: Optional[int],
    moe_skew: float,
    seed: int = 42,
) -> ExpertDistribution:
    """Create an ExpertDistribution from legacy moe_skew parameter."""
    if num_experts is not None and moe_skew > 0.0:
        return make_expert_dist("zipf", moe_skew, num_experts=num_experts, seed=seed)
    return make_expert_dist("uniform", 0.0, seed=seed)


def _fill_allreduce(matrix, groups, volume):
    """AllReduce: each rank exchanges full tensor with every peer in group."""
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    matrix[i][j] += volume


def _fill_alltoall(matrix, groups, total_volume, num_tokens, expert_dist):
    """AllToAll: each src GPU routes tokens via the pluggable ExpertDistribution.

    Each source GPU has num_tokens tokens. The ExpertDistribution determines
    how many tokens each source sends to each destination (e.g. Zipf, uniform,
    or any custom routing model).
    """
    for group in groups:
        bytes_per_token = total_volume / num_tokens if num_tokens > 0 else 0.0
        group_size = len(group)
        for src_local, i in enumerate(group):
            counts = expert_dist.route_tokens(src_local, num_tokens, group_size)
            for idx_j, j in enumerate(group):
                if i != j:
                    matrix[i][j] += float(counts[idx_j] * bytes_per_token)


def _fill_symmetric(matrix, groups, per_rank_bytes):
    """AllGather / ReduceScatter: each rank sends per_rank_bytes to each peer."""
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    matrix[i][j] += per_rank_bytes


def _partition_into_groups(device_ids: List[int], group_size: int) -> List[List[int]]:
    """Split device_ids into contiguous groups of group_size."""
    assert len(device_ids) % group_size == 0
    return [
        device_ids[s : s + group_size]
        for s in range(0, len(device_ids), group_size)
    ]


def _is_moe_template0(cell_schedule) -> bool:
    """Check if cell uses MoE template0 (experts distributed across devices)."""
    cell_name = cell_schedule.cell.get_name()
    if cell_name not in ("MoE", "SwiMoE"):
        return False
    task_dict = cell_schedule.task_mapping.tasks_per_device[0]
    return len(task_dict) < cell_schedule.cell.num_experts


def extract_demand_matrices(
    plan: ExecutionPlan,
    cluster: Cluster,
    model: Transformer,
    act_dtype: DTYPE,
    num_tokens: int,
    moe_skew: float = 0.0,
    expert_dist: Optional[ExpertDistribution] = None,
) -> Dict[str, List[List[float]]]:
    """Extract R×R demand matrices from an execution plan.

    Args:
        plan: The execution plan from search.
        cluster: Full cluster (needed to reconstruct all replica/stage device IDs).
        model: Transformer model (for hidden_size).
        act_dtype: Activation data type (for bytes_per_element).
        num_tokens: Representative total tokens per iteration (across all replicas).
        moe_skew: Zipf exponent for MoE expert popularity (0 = uniform).
            Ignored when expert_dist is provided.
        expert_dist: Pluggable expert routing distribution. If None, one is
            created from moe_skew (ZipfDistribution or UniformDistribution).

    Returns:
        Dict mapping collective name -> R×R matrix.
    """
    R = cluster.get_num_devices()
    hidden_size = model.hidden_size
    bytes_per_elem = act_dtype.size

    parallel_schedule = plan.parallel_schedule
    stage_schedule = parallel_schedule.stage_schedule
    num_model_replicas = parallel_schedule.num_model_replicas
    num_stages = parallel_schedule.num_stages

    # Find attention cell replicas count.
    num_attn_cell_replicas = 1
    for cs in stage_schedule.cell_schedules:
        if cs.cell.is_attn():
            num_attn_cell_replicas = cs.num_replicas
            break

    # Build expert routing distribution (pluggable).
    if expert_dist is None:
        moe_num_experts = None
        for cs in stage_schedule.cell_schedules:
            if cs.cell.get_name() in ("MoE", "SwiMoE"):
                moe_num_experts = cs.cell.num_experts
                break
        expert_dist = _make_expert_dist_legacy(moe_num_experts, moe_skew)

    matrices: Dict[str, List[List[float]]] = {}

    def ensure_matrix(name):
        if name not in matrices:
            matrices[name] = [[0.0] * R for _ in range(R)]
        return matrices[name]

    # Reconstruct device IDs for all replicas and stages.
    replica_clusters = cluster.partition(num_model_replicas)

    for replica_cluster in replica_clusters:
        stage_clusters = replica_cluster.partition(num_stages)

        for stage_cluster in stage_clusters:
            stage_device_ids = get_device_ids(stage_cluster)
            num_stage_devices = len(stage_device_ids)

            # Tokens per stage (total across all cell replicas within this stage).
            num_total_input_tokens = num_tokens // (num_model_replicas * num_stages)

            for i, cell_schedule in enumerate(stage_schedule.cell_schedules):
                num_replicas = cell_schedule.num_replicas
                num_input_tokens = (
                    num_total_input_tokens + num_replicas - 1
                ) // num_replicas

                # MoE template0 token adjustment (simulator.py lines 876-881).
                if _is_moe_template0(cell_schedule):
                    num_devices_in_cell = cell_schedule.task_mapping.get_num_devices()
                    num_input_tokens = max(num_input_tokens // num_devices_in_cell, 1)

                # Process resharding comms.
                for comm in stage_schedule.reshard_comms[i]:
                    num_input_tokens = int(num_input_tokens * comm.size_factor)
                    num_input_tokens = max(num_input_tokens, 1)

                    # Skip 1-device comms (no-ops).
                    if comm.num_devices <= 1:
                        # Still update num_input_tokens for downstream.
                        if comm.comm_type == CommType.AllGather:
                            num_input_tokens *= comm.num_devices
                        elif comm.comm_type == CommType.ReduceScatter:
                            num_input_tokens = max(
                                num_input_tokens // comm.num_devices, 1
                            )
                        continue

                    groups = _partition_into_groups(stage_device_ids, comm.num_devices)

                    if comm.comm_type == CommType.AllReduce:
                        volume = num_input_tokens * hidden_size * bytes_per_elem
                        _fill_allreduce(
                            ensure_matrix("tp_allreduce"), groups, volume
                        )

                    elif comm.comm_type == CommType.AllGather:
                        total_volume = (
                            num_input_tokens
                            * comm.num_devices
                            * hidden_size
                            * bytes_per_elem
                        )
                        per_rank = total_volume // comm.num_devices
                        _fill_symmetric(
                            ensure_matrix("reshard_allgather"), groups, per_rank
                        )
                        num_input_tokens *= comm.num_devices

                    elif comm.comm_type == CommType.ReduceScatter:
                        volume = num_input_tokens * hidden_size * bytes_per_elem
                        per_rank = volume // comm.num_devices
                        _fill_symmetric(
                            ensure_matrix("reshard_reducescatter"), groups, per_rank
                        )
                        num_input_tokens = max(
                            num_input_tokens // comm.num_devices, 1
                        )

                    elif comm.comm_type == CommType.AllToAll:
                        total_volume = num_input_tokens * hidden_size * bytes_per_elem

                        _fill_alltoall(
                            ensure_matrix("ep_alltoall"),
                            groups,
                            total_volume,
                            num_input_tokens,
                            expert_dist,
                        )

                    else:
                        raise NotImplementedError(
                            f"Unsupported comm type: {comm.comm_type}"
                        )

    return matrices


def save_demand_matrices(
    matrices: Dict[str, List[List[float]]],
    output_dir: str,
    prefix: str,
) -> None:
    """Save demand matrices as text files matching demand-matrix/ project format."""
    os.makedirs(output_dir, exist_ok=True)
    for name, matrix in matrices.items():
        path = os.path.join(output_dir, f"{prefix}.{name}.txt")
        with open(path, "w") as f:
            f.write(repr(matrix))
            f.write("\n")
        print(f"  Saved {path} ({len(matrix)}x{len(matrix[0])})")
