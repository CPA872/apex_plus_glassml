"""Modal image with flash-attn + CUDA toolchain for kernel profiling.

Uses a prebuilt flash-attn 2.8.3 wheel for CUDA 13.1 / torch 2.9 / py3.12
from the mjun0812/flash-attention-prebuild-wheels release, which avoids the
20+ minute from-source build flash-attn normally requires on a fresh image.

Targets Blackwell (sm_100, e.g. B200) and keeps Hopper / consumer-Blackwell
in the arch list so the same image runs across the lab's GPU pool.
"""
import modal

FLASH_ATTN_WHEEL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/"
    "download/v0.7.11/"
    "flash_attn-2.8.3%2Bcu131torch2.9-cp312-cp312-"
    "manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
)

flash_attn_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0;10.0;12.0+PTX",
    })
    .pip_install(
        "torch==2.9.*",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
        "pandas",
        "numpy",
        # Official NVIDIA NVML bindings (supersedes the third-party `pynvml`
        # PyPI package; still imported as `import pynvml`).
        "nvidia-ml-py",
    )
    .pip_install(FLASH_ATTN_WHEEL)
)
