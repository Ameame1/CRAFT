#!/bin/bash
# ========================================
# Start CRAFT Judge Server
# Used for computing R_faith reward during GRPO training
# ========================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0

# Resolve python: use $CRAFT_PYTHON_BIN if set, else the repo venv.
CRAFT_PYTHON_BIN="${CRAFT_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin}"
PYTHON="${PYTHON:-$CRAFT_PYTHON_BIN/python}"
export PATH="$CRAFT_PYTHON_BIN:$PATH"

# Default settings
PORT=${PORT:-8000}
CUDA_DEVICES=${CUDA_DEVICES:-"4,5,6,7"}
MODEL=${MODEL:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found or not executable: $PYTHON"
    exit 1
fi

echo "=============================================="
echo "Starting CRAFT Judge Server"
echo "=============================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "CUDA devices: $CUDA_DEVICES"
echo "=============================================="

cd "$PROJECT_ROOT"

"$PYTHON" src/judge/server.py \
    --port $PORT \
    --cuda-devices $CUDA_DEVICES \
    ${MODEL:+--model "$MODEL"}
