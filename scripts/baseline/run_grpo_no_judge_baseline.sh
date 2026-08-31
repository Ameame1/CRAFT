#!/bin/bash
# ========================================
# 1.5B GRPO baseline WITHOUT judge — for judge-overhead analysis
#
# Pairs with the existing 1.5B w/ judge run at:
#   checkpoints/Qwen2.5-1.5B-Instruct-Solver-GRPO/v1-20260114-210118
#   (390 steps, 13.29 hr, 122.71 s/step, on 2 H100)
#
# This run:
#   - Same model, same data, same v1 template, same effective batch (512)
#   - Judge reward REMOVED from reward_funcs
#   - 4 H100 (TP=1 each, ZeRO-2), 50 steps for timing
#   - No judge server needed (no judge calls)
# ========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0

# Defaults: 4 H100s, no judge server
NUM_GPUS=${NUM_GPUS:-4}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
CONFIG_NAME=${CONFIG_NAME:-"grpo_1p5b_no_judge_baseline"}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus|-g)        NUM_GPUS="$2"; shift 2 ;;
        --cuda-devices|-c) CUDA_DEVICES="$2"; shift 2 ;;
        --config|--config-name) CONFIG_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "1.5B GRPO baseline WITHOUT judge"
echo "=============================================="
echo "Project root : $PROJECT_ROOT"
echo "Config       : cfg/${CONFIG_NAME}.yaml"
echo "Num GPUs     : $NUM_GPUS"
echo "CUDA devices : $CUDA_DEVICES"
echo "Note         : NO judge server is required (no judge calls)."
echo "=============================================="

# Sanity-check that this config really has judge removed
if grep -q "judge_overall_consistency_reward" "$PROJECT_ROOT/cfg/${CONFIG_NAME}.yaml" 2>/dev/null; then
    echo "WARNING: $CONFIG_NAME still references judge_overall_consistency_reward."
    echo "         If that's intentional, start the judge server first;"
    echo "         otherwise edit the config to remove it."
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

cd "$PROJECT_ROOT"

# Resolve python/torchrun: use $CRAFT_PYTHON_BIN if set, else the repo venv.
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

# Tag the run output dir with a timestamp so consecutive runs don't overwrite
STAMP=$(date +%Y%m%d-%H%M%S)
export CRAFT_RUN_TAG="$STAMP"

# Run
LOG_FILE="$PROJECT_ROOT/logs/grpo_baseline_${CONFIG_NAME}_${STAMP}.log"
mkdir -p "$PROJECT_ROOT/logs"
echo "Log -> $LOG_FILE"

if [ "$NUM_GPUS" -gt 1 ]; then
    "$TORCHRUN" --nproc_per_node=$NUM_GPUS \
        src/craft/run_grpo.py --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"
else
    "$PYTHON" src/craft/run_grpo.py --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"
fi

echo "=============================================="
echo "Done. Tensorboard: tensorboard --logdir $PROJECT_ROOT/checkpoints/CRAFT-1.5B-NoJudge-Baseline"
echo "Compare per-step time against existing w/ judge run:"
echo "  $PROJECT_ROOT/checkpoints/Qwen2.5-1.5B-Instruct-Solver-GRPO/v1-20260114-210118/runs"
echo "=============================================="
