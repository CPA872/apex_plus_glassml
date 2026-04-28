"""
Dispatcher for APEX+ compute profilers.

Usage:
    uv run python run.py <backend> [backend-args...]

Backends:
    gemm         torch.matmul over (m, k, n) × dtype grid
    mha          xFormers CUTLASS attention (legacy)
    flash_attn   FlashAttention v2/v3 — MHA / GQA / SWA
    flash_mla    FlashMLA — DeepSeek Multi-head Latent Attention (decode)

Pass --help after the backend name to see its arguments:
    uv run python run.py flash_attn --help
"""
import os
import runpy
import sys

BACKENDS = {
    "gemm": "gemm.py",
    "mha": "mha.py",
    "flash_attn": "flash_attn.py",
    "flash_mla": "flash_mla.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 0)

    backend = sys.argv[1]
    if backend not in BACKENDS:
        print(f"Unknown backend: {backend!r}")
        print(f"Available: {', '.join(BACKENDS)}")
        sys.exit(1)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), BACKENDS[backend])
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
