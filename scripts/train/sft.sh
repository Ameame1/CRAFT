#!/bin/bash
# ========================================
# CRAFT SFT Training Script
# Supervised fine-tuning for comparison experiments
# ========================================
# Usage:
#   ./scripts/train/sft.sh                    # Use config defaults (v1)
#   ./scripts/train/sft.sh --version v2       # Train on v2 template
#   ./scripts/train/sft.sh --version v3       # Train on v3 template
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

# Default settings
NUM_GPUS=${NUM_GPUS:-2}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}
TEMPLATE_VERSION=${TEMPLATE_VERSION:-"v1"}
CONFIG_NAME=${CONFIG_NAME:-"sft"}

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
        --config|--config-name)
            CONFIG_NAME="$2"
            shift 2
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

# Set data path based on template version (canonical default; configs may override)
DATA_FILE="data/train/grpo/grpo_25000_${TEMPLATE_VERSION}_messages.jsonl"

echo "=============================================="
echo "CRAFT SFT Training"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Template version: $TEMPLATE_VERSION"
echo "Config name:      $CONFIG_NAME"
echo "Default data file (may be overridden by cfg): $DATA_FILE"
echo "Number of GPUs:   $NUM_GPUS"
echo "CUDA devices:     $CUDA_DEVICES"
echo "=============================================="

# Only check the default data file when running the default 'sft' config; for
# custom configs (e.g. sft_rscraft) the real dataset is set inside the YAML.
if [ "$CONFIG_NAME" = "sft" ] && [ ! -f "$PROJECT_ROOT/$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export CRAFT_TEMPLATE_VERSION=$TEMPLATE_VERSION
export CRAFT_DATASET=$DATA_FILE

# Run training
cd "$PROJECT_ROOT"

CRAFT_PYTHON_BIN="${CRAFT_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin}"
PYTHON="${PYTHON:-$CRAFT_PYTHON_BIN/python}"
TORCHRUN="${TORCHRUN:-$CRAFT_PYTHON_BIN/torchrun}"
export PATH="$CRAFT_PYTHON_BIN:$PATH"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found or not executable: $PYTHON"
    exit 1
fi
if [ "$NUM_GPUS" -gt 1 ] && [ ! -x "$TORCHRUN" ]; then
    echo "ERROR: torchrun not found or not executable: $TORCHRUN"
    exit 1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with DeepSpeed
    "$TORCHRUN" --nproc_per_node=$NUM_GPUS \
        src/craft/run_sft.py \
        --config-name "$CONFIG_NAME"
else
    # Single GPU training
    "$PYTHON" src/craft/run_sft.py \
        --config-name "$CONFIG_NAME"
fi

echo "=============================================="
echo "SFT Training Complete"
echo "=============================================="
