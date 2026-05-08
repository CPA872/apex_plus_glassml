#!/bin/bash
# Three-way comparison: NVL64 vs 8x8 IB vs Spectra (single-tier OCS over 64 GPUs).
# Same model / prompt / EP across all three. Energy/spectra config picked up
# from config.yaml in this directory.
set -e

MODEL="deepseek-v3"
GPUS=64
PROMPT=16384
OUTPUT=1
REQS=4096
EP=64
PLANES=4

cd "$(dirname "$0")"

echo "=== NVL64 (single-node, 64 GPUs) ==="
uv run python main.py --model $MODEL --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu H100-SXM-80GB --prompt-len $PROMPT --output-len $OUTPUT \
  --num-requests $REQS --force-ep $EP --frequency 1980 \
  --interconnect nvlink

echo ""
echo "=== 8x8 IB (8 nodes, 8 GPUs each) ==="
uv run python main.py --model $MODEL --num-nodes 8 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --prompt-len $PROMPT --output-len $OUTPUT \
  --num-requests $REQS --force-ep $EP --frequency 1980 \
  --interconnect ib

echo ""
echo "=== Spectra (single-tier OCS, $GPUS GPUs, $PLANES planes) ==="
uv run python main.py --model $MODEL --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu H100-SXM-80GB --prompt-len $PROMPT --output-len $OUTPUT \
  --num-requests $REQS --force-ep $EP --frequency 1980 \
  --interconnect spectra --num-planes $PLANES
