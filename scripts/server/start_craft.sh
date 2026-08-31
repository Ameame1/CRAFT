#!/bin/bash
# ========================================
# Start CRAFT Inference Server
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
PORT=${PORT:-8002}
CUDA_DEVICES=${CUDA_DEVICES:-"0"}
MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found or not executable: $PYTHON"
    exit 1
fi

echo "=============================================="
echo "Starting CRAFT Server"
echo "=============================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "CUDA devices: $CUDA_DEVICES"
echo "=============================================="

cd "$PROJECT_ROOT"

"$PYTHON" src/craft/server.py \
    --port $PORT \
    --cuda-devices $CUDA_DEVICES \
    ${MODEL:+--model "$MODEL"}
