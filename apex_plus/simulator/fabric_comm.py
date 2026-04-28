"""Fabric-level communication events for per-step spectra scheduling.

A fabric event is one scheduling instant on the optical fabric. NCCL group
invocations that fire concurrently for the same logical step (i.e., DP
replicas executing the same layer's same comm) are merged into one event,
since they share the optical fabric in time. Pipeline stages are NOT merged
(in steady state, different stages run different layers concurrently with
different demand matrices; merging would conflate them).

Each event carries everything needed to:
  - build an R x R demand matrix (bytes from src to dst), and
  - hand it to the spectra solver as a single scheduling input.

Output of `iter_fabric_events` is a stream of plain dicts; downstream code
calls `build_demand_matrix(event, R, expert_dist)` to produce the ndarray.
"""

from typing import Dict, Iterator, List, Optional

import numpy as np

from apex_plus.cluster.cluster import Cluster
from apex_plus.execution.plan import ExecutionPlan
from apex_plus.ir.transformer import Transformer
from apex_plus.parallel.comm import CommType
from apex_plus.simulator.demand_matrix import (
    _is_moe_template0,
    _partition_into_groups,
    get_device_ids,
)
from apex_plus.simulator.expert_distribution import ExpertDistribution
from apex_plus.utils.dtype import DTYPE


def _infer_module(comm_type, prev_cell: str, next_cell: str) -> str:
    if comm_type == CommType.AllToAll:
        if next_cell in ("MoE", "SwiMoE"):
            return "A2A_dispatch"
        if prev_cell in ("MoE", "SwiMoE"):
            return "A2A_combine"
        return "A2A"
    if comm_type == CommType.AllReduce:
        return "TP_AllReduce"
    if comm_type == CommType.AllGather:
        return "AllGather"
    if comm_type == CommType.ReduceScatter:
        return "ReduceScatter"
    return str(comm_type).split(".")[-1]


def iter_fabric_events(
    plan: ExecutionPlan,
    cluster: Cluster,
    model: Transformer,
    act_dtype: DTYPE,
    num_tokens: int,
) -> Iterator[Dict]:
    """Walk the plan and yield one fabric event per (stage, layer, cell, comm).

    Concurrent DP replicas are merged into the event's `groups` field. No-op
    comms (`num_devices <= 1`) are skipped, but their `size_factor` still
    advances the running token count for downstream comms in the same block.

    Yields:
        dict with keys: step_id, stage_idx, layer_idx, cell_idx, comm_idx,
        comm_type (CommType), module (str), prev_cell, next_cell,
        groups (tuple[tuple[int, ...], ...]), num_devices, size_factor,
        num_tokens_per_rank, bytes_per_token.
    """
    parallel_schedule = plan.parallel_schedule
    stage_schedule = parallel_schedule.stage_schedule
    num_replicas = parallel_schedule.num_model_replicas
    num_stages = parallel_schedule.num_stages
    num_blocks = stage_schedule.num_blocks
    cell_schedules = stage_schedule.cell_schedules
    reshard_comms = stage_schedule.reshard_comms

    hidden_size = model.hidden_size
    bytes_per_elem = act_dtype.size

    replica_clusters = cluster.partition(num_replicas)

    # Per-stage device IDs are identical across replicas in shape, but use
    # different physical ranks. Collect them once per (replica, stage).
    stage_dev_ids_by_rep: List[List[List[int]]] = []
    for replica_cluster in replica_clusters:
        per_rep = []
        for stage_cluster in replica_cluster.partition(num_stages):
            per_rep.append(get_device_ids(stage_cluster))
        stage_dev_ids_by_rep.append(per_rep)

    step_id = 0
    for stage_idx in range(num_stages):
        # Tokens per stage (total across all cell replicas within this stage).
        num_total_input_tokens = num_tokens // (num_replicas * num_stages)

        for block_idx in range(num_blocks):
            for cell_idx, cell_schedule in enumerate(cell_schedules):
                num_cell_replicas = cell_schedule.num_replicas
                num_input_tokens = (
                    num_total_input_tokens + num_cell_replicas - 1
                ) // num_cell_replicas

                if _is_moe_template0(cell_schedule):
                    num_devices_in_cell = cell_schedule.task_mapping.get_num_devices()
                    num_input_tokens = max(num_input_tokens // num_devices_in_cell, 1)

                cell_name = cell_schedule.cell.get_name()
                # prev_cell wraps within the block loop in steady state.
                prev_cell_name = (
                    cell_schedules[cell_idx - 1].cell.get_name()
                    if cell_idx > 0
                    else cell_schedules[-1].cell.get_name()
                )

                for comm_idx, comm in enumerate(reshard_comms[cell_idx]):
                    num_input_tokens = int(num_input_tokens * comm.size_factor)
                    num_input_tokens = max(num_input_tokens, 1)

                    if comm.num_devices <= 1:
                        # Token-count side-effects (AG inflates, RS deflates).
                        if comm.comm_type == CommType.AllGather:
                            num_input_tokens *= comm.num_devices
                        elif comm.comm_type == CommType.ReduceScatter:
                            num_input_tokens = max(
                                num_input_tokens // comm.num_devices, 1
                            )
                        continue

                    # Merge concurrent DP groups: one group per replica.
                    groups: List[List[int]] = []
                    for r_idx in range(num_replicas):
                        stage_dev_ids = stage_dev_ids_by_rep[r_idx][stage_idx]
                        groups.extend(
                            _partition_into_groups(stage_dev_ids, comm.num_devices)
                        )

                    step_id += 1
                    yield {
                        "step_id": step_id,
                        "stage_idx": stage_idx,
                        "layer_idx": block_idx,
                        "cell_idx": cell_idx,
                        "comm_idx": comm_idx,
                        "comm_type": comm.comm_type,
                        "module": _infer_module(
                            comm.comm_type, prev_cell_name, cell_name
                        ),
                        "prev_cell": prev_cell_name,
                        "next_cell": cell_name,
                        "groups": tuple(tuple(g) for g in groups),
                        "num_devices": comm.num_devices,
                        "size_factor": comm.size_factor,
                        "num_tokens_per_rank": num_input_tokens,
                        "bytes_per_token": float(hidden_size * bytes_per_elem),
                    }

                    # Token-count side-effects after the comm.
                    if comm.comm_type == CommType.AllGather:
                        num_input_tokens *= comm.num_devices
                    elif comm.comm_type == CommType.ReduceScatter:
                        num_input_tokens = max(
                            num_input_tokens // comm.num_devices, 1
                        )


def build_demand_matrix(
    event: Dict,
    R: int,
    expert_dist: Optional[ExpertDistribution] = None,
) -> np.ndarray:
    """Construct an R x R demand matrix (bytes) for a single fabric event.

    Non-participating ranks have zero traffic. For AllToAll, `expert_dist`
    drives per-source token routing (rows differ across sources). For ring
    collectives, traffic is uniform within each group.
    """
    D = np.zeros((R, R), dtype=np.float64)
    comm_type = event["comm_type"]
    groups = event["groups"]
    num_tokens = event["num_tokens_per_rank"]
    bytes_per_token = event["bytes_per_token"]
    num_devices = event["num_devices"]

    if comm_type == CommType.AllToAll:
        if expert_dist is None:
            raise ValueError("AllToAll events require an expert_dist.")
        for group in groups:
            group_size = len(group)
            for src_local, i in enumerate(group):
                counts = expert_dist.route_tokens(src_local, num_tokens, group_size)
                for idx_j, j in enumerate(group):
                    if i != j:
                        D[i, j] += float(counts[idx_j]) * bytes_per_token

    elif comm_type == CommType.AllReduce:
        # Each rank exchanges full tensor with every peer in group.
        volume = num_tokens * bytes_per_token
        for group in groups:
            for i in group:
                for j in group:
                    if i != j:
                        D[i, j] += volume

    elif comm_type in (CommType.AllGather, CommType.ReduceScatter):
        # Per-rank shard goes to every peer.
        if comm_type == CommType.AllGather:
            total_volume = num_tokens * num_devices * bytes_per_token
        else:  # ReduceScatter
            total_volume = num_tokens * bytes_per_token
        per_rank = total_volume // num_devices
        for group in groups:
            for i in group:
                for j in group:
                    if i != j:
                        D[i, j] += per_rank

    else:
        raise NotImplementedError(f"Unsupported comm type: {comm_type}")

    return D
