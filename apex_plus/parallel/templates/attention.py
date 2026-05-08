from typing import Dict, List, Optional

from apex_plus.ir.cells.attention import MLA, MQA
from apex_plus.ir.task import Task
from apex_plus.parallel.comm import CollectiveComm
from apex_plus.parallel.task_parallel import ParallelTemplate, TaskMapping
from apex_plus.parallel.templates.default import DefaultTemplate


class MQATemplate0(ParallelTemplate):

    @staticmethod
    def map_tasks(cell: MQA, num_devices: int) -> Optional[TaskMapping]:
        if cell.num_query_heads < num_devices:
            # Not enough heads to distribute.
            return None

        # Evenly distribute heads to devices, while co-locating the query heads
        # that correspond to the same key and value heads.
        if cell.num_kv_heads < num_devices:
            # KV heads are replicated across devices: each KV head is
            # assigned to its own contiguous slice of devices, and that KV
            # head's query heads are split across that slice.
            num_devices_per_kv_head = [
                num_devices // cell.num_kv_heads for _ in range(cell.num_kv_heads)
            ]
            for i in range(num_devices % cell.num_kv_heads):
                num_devices_per_kv_head[i] += 1

            query_heads_per_device: List[Dict[str, List[Task]]] = []
            for i in range(num_devices):
                query_heads_per_device.append({})

            device_offset = 0
            for i in range(cell.num_kv_heads):
                n_dev = num_devices_per_kv_head[i]
                num_query_heads_per_device = [
                    cell.num_query_per_kv // n_dev for _ in range(n_dev)
                ]
                for j in range(cell.num_query_per_kv % n_dev):
                    num_query_heads_per_device[j] += 1

                head_start = i * cell.num_query_per_kv
                for j in range(n_dev):
                    head_end = head_start + num_query_heads_per_device[j]
                    query_heads_per_device[device_offset + j].setdefault(
                        "MQAHead", []
                    ).extend(cell.heads[head_start:head_end])
                    head_start = head_end
                device_offset += n_dev
        else:
            # Distibute KV heads to devices as evenly as possible.
            num_kv_heads_per_device = [
                cell.num_kv_heads // num_devices for _ in range(num_devices)
            ]
            for i in range(cell.num_kv_heads % num_devices):
                num_kv_heads_per_device[i] += 1

            query_heads_per_device: List[Dict[str, List[Task]]] = []
            for i in range(num_devices):
                query_heads_per_device.append({})
            start = 0
            for i in range(num_devices):
                end = start + num_kv_heads_per_device[i]
                query_heads_per_device[i].setdefault("MQAHead", []).extend(
                    cell.heads[
                        start * cell.num_query_per_kv : end * cell.num_query_per_kv
                    ]
                )
                start = end

        task_mapping = TaskMapping(
            query_heads_per_device,
            CollectiveComm(comm_type="AllReduce", num_devices=num_devices),
        )
        return task_mapping


class MLATemplate0(ParallelTemplate):
    """Megatron-style TP for MLA: distribute query heads across devices,
    AllReduce close. KV is a single shared latent (H_KV=1) so there's no
    KV-head partitioning logic — every device replicates the kv_lora and
    qk_rope state from the shared KV-down projection.
    """

    @staticmethod
    def map_tasks(cell: MLA, num_devices: int) -> Optional[TaskMapping]:
        if cell.num_query_heads < num_devices:
            return None

        heads_per_device = [
            cell.num_query_heads // num_devices for _ in range(num_devices)
        ]
        for i in range(cell.num_query_heads % num_devices):
            heads_per_device[i] += 1

        tasks_per_device: List[Dict[str, List[Task]]] = []
        start = 0
        for i in range(num_devices):
            end = start + heads_per_device[i]
            tasks_per_device.append({"MLAHead": cell.heads[start:end]})
            start = end

        return TaskMapping(
            tasks_per_device,
            CollectiveComm(comm_type="AllReduce", num_devices=num_devices),
        )


# Cell name -> list of templates.
ATTENTION_TEMPLATES_REGISTRY = {
    "MHA": [DefaultTemplate],
    "BiMHA": [DefaultTemplate],
    "MQA": [MQATemplate0],
    "MLA": [MLATemplate0],
    "ParallelMHAMLP": [DefaultTemplate],
}
