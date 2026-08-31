#!/bin/bash
# CRAFT Data Preprocessing Pipeline
#
# This script runs the full preprocessing pipeline:
#   1. Download datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue)
#   2. Generate GRPO training data for all template versions (v1-v5)
#
# Usage:
#   ./scripts/preprocess/process.sh [options]
#
# Options:
#   --skip-download    Skip dataset download (use existing raw data)
#   --force-download   Force re-download even if files exist
#   --samples N        Number of samples to generate (default: 25000)
#   --versions V1 V2   Template versions to generate (default: v1 v2 v3 v4 v5)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default options
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false
SAMPLES=25000
VERSIONS="v1 v2 v3 v4 v5"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --versions)
            shift
            VERSIONS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                VERSIONS="${VERSIONS} $1"
                shift
            done
            VERSIONS=$(echo "$VERSIONS" | xargs)  # trim whitespace
            ;;
        -h|--help)
            echo "CRAFT Data Preprocessing Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-download    Skip dataset download (use existing raw data)"
            echo "  --force-download   Force re-download even if files exist"
            echo "  --samples N        Number of samples to generate (default: 25000)"
            echo "  --versions V1 V2   Template versions to generate (default: v1 v2 v3 v4 v5)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate replay 2>/dev/null || true
fi

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "========================================"
echo "CRAFT Data Preprocessing Pipeline"
echo "========================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Samples: ${SAMPLES}"
echo "Versions: ${VERSIONS}"
echo ""

cd "${PROJECT_ROOT}"

# Step 1: Download datasets
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "========================================"
    echo "Step 1: Download Datasets"
    echo "========================================"

    DOWNLOAD_ARGS=""
    if [ "$FORCE_DOWNLOAD" = true ]; then
        DOWNLOAD_ARGS="--force"
    fi

    python scripts/preprocess/download_datasets.py $DOWNLOAD_ARGS
    echo ""
fi

# Step 2: Generate GRPO training data
echo "========================================"
echo "Step 2: Generate GRPO Training Data"
echo "========================================"

python scripts/preprocess/generate_grpo_data.py \
    --samples "$SAMPLES" \
    --versions $VERSIONS

echo ""
echo "========================================"
echo "Preprocessing Complete!"
echo "========================================"
echo ""
echo "Output files in: ${PROJECT_ROOT}/data/data_train/grpo/"
ls -la "${PROJECT_ROOT}/data/data_train/grpo/" 2>/dev/null || echo "  (directory not yet created)"
