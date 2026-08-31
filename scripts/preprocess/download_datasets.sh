#!/bin/bash
# Download CRAFT training datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue)
#
# Usage:
#   ./scripts/preprocess/download_datasets.sh [--force]
#
# The datasets will be downloaded to CRAFT/data/raw/{dataset_name}/

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate replay 2>/dev/null || true
fi

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "CRAFT Data Download Script"
echo "=========================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Run the download script
cd "${PROJECT_ROOT}"
python scripts/preprocess/download_datasets.py "$@"
