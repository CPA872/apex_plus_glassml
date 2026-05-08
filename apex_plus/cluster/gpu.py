import enum

from apex_plus.cluster.device import Device
from apex_plus.utils.dtype import DTYPE

_GB = 1 << 30


class GPUType(enum.Enum):
    V100_PCIE_16GB = enum.auto()
    H100_SXM_80GB = enum.auto()
    H200_SXM_141GB = enum.auto()
    B200_SXM_192GB = enum.auto()


_GPU_REGISTRY = {
    "V100-PCIE-16GB": GPUType.V100_PCIE_16GB,
    "H100-SXM-80GB": GPUType.H100_SXM_80GB,
    "H200-SXM-141GB": GPUType.H200_SXM_141GB,
    "B200-SXM-192GB": GPUType.B200_SXM_192GB,
}

_GPU_TYPE_TO_MEMORY_GB = {
    GPUType.V100_PCIE_16GB: 16 * _GB,
    GPUType.H100_SXM_80GB: 80 * _GB,
    GPUType.H200_SXM_141GB: 141 * _GB,
    GPUType.B200_SXM_192GB: 192 * _GB,
}

_GPU_TYPE_TO_TOPOLOGY = {
    GPUType.V100_PCIE_16GB: "pcie",
    GPUType.H100_SXM_80GB: "nvlink",
    GPUType.H200_SXM_141GB: "nvlink",
    GPUType.B200_SXM_192GB: "nvlink",
}

# Tensor-Core peak FLOPS (dense, no sparsity). BF16 == FP16 throughput on
# Hopper/Blackwell. V100 (Volta) does not support BF16 natively.
_GPU_PEAK_FLOPS = {
    GPUType.V100_PCIE_16GB: {
        DTYPE.FLOAT32: 7e12,
        DTYPE.FLOAT16: 14e12,
    },
    GPUType.H100_SXM_80GB: {
        DTYPE.FLOAT32: 67e12,
        DTYPE.FLOAT16: 1979e12,
        DTYPE.BFLOAT16: 1979e12,
        DTYPE.FLOAT8: 3958e12,
    },
    GPUType.H200_SXM_141GB: {
        DTYPE.FLOAT32: 67e12,
        DTYPE.FLOAT16: 1979e12,
        DTYPE.BFLOAT16: 1979e12,
        DTYPE.FLOAT8: 3958e12,
    },
    GPUType.B200_SXM_192GB: {
        DTYPE.FLOAT32: 80e12,
        DTYPE.FLOAT16: 4500e12,
        DTYPE.BFLOAT16: 4500e12,
        DTYPE.FLOAT8: 9000e12,
    },
}

_GPU_PEAK_MEM_BANDWIDTH = {
    GPUType.V100_PCIE_16GB: 900e9,
    GPUType.H100_SXM_80GB: 3.35e12,
    GPUType.H200_SXM_141GB: 4.8e12,
    GPUType.B200_SXM_192GB: 8.0e12,
}


class GPU(Device):

    def __init__(
        self,
        device_id: int,
        device_type: str,
    ) -> None:
        self.device_id = device_id
        device_type = device_type.upper()
        self.device_type = device_type

        if device_type not in _GPU_REGISTRY:
            raise ValueError(f"Unknown GPU: {device_type}")
        self.gpu_type = _GPU_REGISTRY[device_type]
        self.total_memory = _GPU_TYPE_TO_MEMORY_GB[self.gpu_type]
        self.peak_flops = _GPU_PEAK_FLOPS[self.gpu_type]
        self.peak_mem_bandwidth = _GPU_PEAK_MEM_BANDWIDTH[self.gpu_type]

    def get_memory_capacity(self) -> int:
        return self.total_memory

    def __repr__(self) -> str:
        return (
            f"GPU(device_id={self.device_id}, "
            f"device_type={self.device_type},"
            f"peak_flops={self.peak_flops},"
            f"peak_mem_bandwidth={self.peak_mem_bandwidth},"
            f"device_type={self.device_type},"
        )

    @staticmethod
    def get_topology(gpu: str) -> str:
        gpu_type = _GPU_REGISTRY[gpu.upper()]
        return _GPU_TYPE_TO_TOPOLOGY[gpu_type]
