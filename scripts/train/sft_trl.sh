#!/bin/bash
# RS-CRAFT 7B SFT via TRL + accelerate + DeepSpeed ZeRO-2.
# Mirrors the paper SFT recipe (CRAFT_package/scripts/run_sft.sh).
#
# Pre-req: select_best.py output at data/train/rscraft/rscraft_best_of_4_v1.jsonl
# Output dir: checkpoints/RS-CRAFT-7B-v1-trl
#
# Usage:
#   NUM_GPUS=4 CUDA_DEVICES=0,1,2,3 ./scripts/train/sft_trl.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Resolve env
CRAFT_PYTHON_BIN="${CRAFT_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin}"
ACCELERATE="${ACCELERATE:-$CRAFT_PYTHON_BIN/accelerate}"
export PATH="$CRAFT_PYTHON_BIN:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONNOUSERSITE=1

if [ ! -x "$ACCELERATE" ]; then
    echo "ERROR: accelerate not found or not executable: $ACCELERATE"
    exit 1
fi

# Defaults
NUM_GPUS=${NUM_GPUS:-4}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
DATASET=${DATASET:-"data/train/rscraft/rscraft_best_of_4_v1.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/RS-CRAFT-7B-v1-trl"}
ACCEL_CFG=${ACCEL_CFG:-"cfg/accelerate_deepspeed_zero2.yaml"}

# Hyperparameters (paper SFT recipe — Appendix A)
LR=${LR:-"2e-5"}
EPOCHS=${EPOCHS:-3}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}    # effective batch = 2 * 16 * 4 = 128
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}    # > p99 of input+output tokens (3.5K), so no truncation

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0
export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

echo "=============================================="
echo "RS-CRAFT 7B SFT — TRL + accelerate + DeepSpeed"
echo "=============================================="
echo "Model:        $MODEL"
echo "Dataset:      $DATASET"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS  ($CUDA_DEVICES)"
echo "LR / epochs:  $LR / $EPOCHS"
echo "Batch:        $PER_DEVICE_BATCH per GPU * $GRAD_ACCUM accum * $NUM_GPUS GPUs = $((PER_DEVICE_BATCH*GRAD_ACCUM*NUM_GPUS))"
echo "Max seq:      $MAX_SEQ_LEN"
echo "=============================================="

if [ ! -f "$PROJECT_ROOT/$DATASET" ]; then
    echo "ERROR: dataset not found: $DATASET"
    exit 1
fi

"$ACCELERATE" launch \
    --config_file "$ACCEL_CFG" \
    --num_processes "$NUM_GPUS" \
    scripts/train/sft_trl.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATASET" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_SEQ_LEN" \
    --bf16 True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.3 \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard

echo "=============================================="
echo "SFT complete -> $OUTPUT_DIR"
echo "=============================================="
