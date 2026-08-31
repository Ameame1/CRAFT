#!/bin/bash
# ========================================
# CRAFT Evaluation Script
# ========================================
# Usage:
#   ./scripts/eval/eval.sh                                    # Use config defaults
#   ./scripts/eval/eval.sh --version v1 --dataset hotpotqa   # Specific version/dataset
#   ./scripts/eval/eval.sh --version v2 --dataset musique    # v2 on MuSiQue
#   ./scripts/eval/eval.sh --version v3 --dataset 2wiki      # v3 on 2WikiMultiHop
# ========================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0

# Default settings
CUDA_DEVICES=${CUDA_DEVICES:-"0"}
TEMPLATE_VERSION=${TEMPLATE_VERSION:-"v1"}
DATASET=${DATASET:-"hotpotqa"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v)
            TEMPLATE_VERSION="$2"
            shift 2
            ;;
        --dataset|-d)
            DATASET="$2"
            shift 2
            ;;
        --cuda-devices|-c)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
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

# Validate dataset
case "$DATASET" in
    hotpotqa|musique|2wiki)
        ;;
    *)
        echo "ERROR: Invalid dataset: $DATASET"
        echo "Valid options: hotpotqa, musique, 2wiki"
        exit 1
        ;;
esac

# Set data path based on template version and dataset
DATA_FILE="data/test/${DATASET}_2000_${TEMPLATE_VERSION}.jsonl"

echo "=============================================="
echo "CRAFT Evaluation"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Template version: $TEMPLATE_VERSION"
echo "Dataset: $DATASET"
echo "Data file: $DATA_FILE"
echo "CUDA devices: $CUDA_DEVICES"
if [ -n "$MODEL" ]; then
    echo "Model: $MODEL"
fi
echo "=============================================="

# Check if data file exists
if [ ! -f "$PROJECT_ROOT/$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export CRAFT_TEMPLATE_VERSION=$TEMPLATE_VERSION
export CRAFT_DATASET=$DATA_FILE

cd "$PROJECT_ROOT"

# Build command
CMD="python src/craft/run_eval.py --dataset $DATA_FILE --template_version $TEMPLATE_VERSION"
if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

eval $CMD

echo "=============================================="
echo "Evaluation Complete"
echo "=============================================="
