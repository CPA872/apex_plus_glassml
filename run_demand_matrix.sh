#!/bin/bash
# Extract demand matrices for various models and parallelism configs
set -e

OUT="../profiling/demand_matrix"
mkdir -p "$OUT"

echo "=== DeepSeek-V3, TP1 x EP64 (uniform) ==="
uv run python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --demand-matrix "$OUT" --dm-num-tokens 8192 --frequency 1980


echo ""
echo "=== DeepSeek-V3, TP1 x EP64 (Zipf skew=1.0) ==="
uv run python main.py --model deepseek-v3 --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 8192 --output-len 1 --num-requests 4096 \
  --force-ep 64 --demand-matrix "$OUT" --dm-num-tokens 8192 --moe-skew 1.0 --frequency 1980


echo ""
echo "=== LLaMA 3.1 405B, TP-only (dense model) ==="
uv run python main.py --model llama3.1-405b --num-nodes 1 --num-gpus-per-node 64 \
  --gpu H100-SXM-80GB --prompt-len 2048 --output-len 128 --num-requests 64 \
  --demand-matrix "$OUT" --dm-num-tokens 2048 --frequency 1980

echo ""
echo "Output files:"
ls -lh "$OUT"/*.txt
