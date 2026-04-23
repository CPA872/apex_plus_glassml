#!/bin/bash
# Compare NVL64 vs 8x8 IB for DeepSeek-V3 (TP1 x EP64, training-style)
set -e

MODEL="deepseek-v3"
GPUS=64
PROMPT=16384
OUTPUT=1
REQS=4096
EP=64

echo "=== NVL64 (single-node, 64 GPUs) ==="
uv run python main.py --model $MODEL --num-nodes 1 --num-gpus-per-node $GPUS \
  --gpu H100-SXM-80GB --prompt-len $PROMPT --output-len $OUTPUT \
  --num-requests $REQS --force-ep $EP --frequency 1980 --energy-config /mnt/alpha/yuepan/glassml/apex_plus/energy_config.yaml

echo ""
echo "=== 8x8 IB (8 nodes, 8 GPUs each) ==="
uv run python main.py --model $MODEL --num-nodes 8 --num-gpus-per-node 8 \
  --gpu H100-SXM-80GB --prompt-len $PROMPT --output-len $OUTPUT \
  --num-requests $REQS --force-ep $EP --frequency 1980 --energy-config /mnt/alpha/yuepan/glassml/apex_plus/energy_config.yaml
