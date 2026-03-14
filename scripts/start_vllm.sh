#!/usr/bin/env bash
# Start vLLM server for Qwen2.5-3B-Instruct on RTX 4060 Ti 16GB
set -euo pipefail

# Ensure CUDA is visible
export CUDA_VISIBLE_DEVICES=0

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

MODEL="${PDD_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

echo "Starting vLLM server with ${MODEL} ..."
echo "  GPU memory utilization: 0.80"
echo "  Max concurrent sequences: 32"
echo "  Max model length: 2048"
echo "  Port: 8000"

vllm serve "$MODEL" \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 32 \
    --max-model-len 2048 \
    --port 8000
