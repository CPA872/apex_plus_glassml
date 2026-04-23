#!/bin/bash
# Run all model benchmarks on NVL64 (64 H100 GPUs, single-node)
set -e

GPUS=64
GPU_TYPE="H100-SXM-80GB"

echo "=== LLaMA 3.1 405B (dense, TP) ==="
uv run python main.py --model llama3.1-405b --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu $GPU_TYPE --prompt-len 2048 --output-len 128 --num-requests 64

echo ""
echo "=== 64-Expert MoE, large experts (optimizer-chosen plan) ==="
uv run python main.py --model moe-64x-large --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu $GPU_TYPE --prompt-len 8192 --output-len 1 --num-requests 4096

echo ""
echo "=== DeepSeek-V3, 256 experts (forced TP1 x EP64) ==="
uv run python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu $GPU_TYPE --prompt-len 8192 --output-len 1 --num-requests 4096 --force-ep 64

echo ""
echo "=== DeepSeek-V2, 160 experts (forced TP1 x EP64) ==="
uv run python main.py --model deepseek-v2 --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu $GPU_TYPE --prompt-len 8192 --output-len 1 --num-requests 4096 --force-ep 64
