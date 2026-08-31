#!/bin/bash
# ========================================
# CRAFT GRPO Training Script
# Group Relative Policy Optimization with Judge Rewards
# ========================================
# Prerequisites:
# 1. Start judge server first: scripts/server/start_judge.sh
# 2. Training data available in data/train/
# ========================================
# Usage:
#   ./scripts/train/grpo.sh                    # Use config defaults (v1)
#   ./scripts/train/grpo.sh --version v2       # Train on v2 template
#   ./scripts/train/grpo.sh --version v3       # Train on v3 template
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
# Reduce allocator fragmentation in long-context runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Default settings
NUM_GPUS=${NUM_GPUS:-2}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}
JUDGE_URL=${JUDGE_URL:-"http://localhost:8000/v1"}
TEMPLATE_VERSION=${TEMPLATE_VERSION:-"v1"}
CONFIG_NAME=${CONFIG_NAME:-"grpo"}
SKIP_JUDGE_CHECK=${SKIP_JUDGE_CHECK:-0}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v)
            TEMPLATE_VERSION="$2"
            shift 2
            ;;
        --gpus|-g)
            NUM_GPUS="$2"
            shift 2
            ;;
        --cuda-devices|-c)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --judge-url|-j)
            JUDGE_URL="$2"
            shift 2
            ;;
        --config|--config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --skip-judge-check)
            SKIP_JUDGE_CHECK=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate template version
if [[ ! "$TEMPLATE_VERSION" =~ ^v[1-5]$ ]]; then
    echo "ERROR: Invalid template version: $TEMPLATE_VERSION"
    echo "Valid options: v1, v2, v3, v4, v5"
    exit 1
fi

# Read the dataset path from the selected YAML configuration.
CONFIG_FILE="$PROJECT_ROOT/cfg/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
DATA_FILE=$(awk '/^dataset:/ {print $2; exit}' "$CONFIG_FILE" | tr -d "\"'")
if [ -z "$DATA_FILE" ]; then
    echo "ERROR: No dataset entry found in $CONFIG_FILE"
    exit 1
fi

echo "=============================================="
echo "CRAFT GRPO Training"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Template version: $TEMPLATE_VERSION"
echo "Data file: $DATA_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA devices: $CUDA_DEVICES"
echo "Judge URL: $JUDGE_URL"
echo "=============================================="

# Check if data file exists
if [ ! -f "$PROJECT_ROOT/$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Check the judge only when the selected reward set uses it.
if ! grep -q "judge_overall_consistency_reward" "$CONFIG_FILE"; then
    echo "Selected config has no judge reward; skipping judge server check"
elif [ "$SKIP_JUDGE_CHECK" = "1" ]; then
    echo "Skipping judge server check (--skip-judge-check)"
else
    echo "Checking judge server..."
    if curl -s "$JUDGE_URL/models" > /dev/null 2>&1; then
        echo "Judge server is running at $JUDGE_URL"
    else
        echo "ERROR: Judge server is not running at $JUDGE_URL"
        echo "Please start the judge server first:"
        echo "  ./scripts/server/start_judge.sh"
        exit 1
    fi
fi

# Set environment
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export LOCAL_JUDGE_URL=$JUDGE_URL
export CRAFT_TEMPLATE_VERSION=$TEMPLATE_VERSION
export CRAFT_DATASET=$DATA_FILE

# Resolve python/torchrun: use $CRAFT_PYTHON_BIN if set, else the repo venv.
CRAFT_PYTHON_BIN="${CRAFT_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin}"
PYTHON="${PYTHON:-$CRAFT_PYTHON_BIN/python}"
TORCHRUN="${TORCHRUN:-$CRAFT_PYTHON_BIN/torchrun}"
export PATH="$CRAFT_PYTHON_BIN:$PATH"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found or not executable: $PYTHON"
    exit 1
fi
# Run training
cd "$PROJECT_ROOT"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Using the selected interpreter keeps distributed workers in the same env.
    "$PYTHON" -m torch.distributed.run --nproc_per_node=$NUM_GPUS \
        src/craft/run_grpo.py \
        --config-name "$CONFIG_NAME"
else
    # Single GPU training
    "$PYTHON" src/craft/run_grpo.py \
        --config-name "$CONFIG_NAME"
fi

echo "=============================================="
echo "GRPO Training Complete"
echo "=============================================="
