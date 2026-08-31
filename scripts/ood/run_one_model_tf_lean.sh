#!/bin/bash
# Args: $1=model_path  $2=tag  $3=space-separated datasets
set -e
PYTHON="${PYTHON:-python}"
MODEL_PATH="$1"
MODEL_TAG="$2"
DATASETS="$3"  # e.g. "multihoprag" or "hotpotqa musique 2wiki multihoprag"

RESULTS_OUT="data/results/tf/${MODEL_TAG}"
mkdir -p "$RESULTS_OUT"
LOG_DIR="/tmp/tf-${MODEL_TAG}"
mkdir -p "$LOG_DIR"

echo "==== Booting server for ${MODEL_TAG} (datasets: ${DATASETS}) ===="
> "$LOG_DIR/server.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_TRITON_FLASH_ATTN=0 \
nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --port 8002 --host 0.0.0.0 --dtype bfloat16 \
  --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
until curl -sf http://localhost:8002/v1/models > /dev/null 2>&1 || ! kill -0 $SERVER_PID 2>/dev/null; do sleep 8; done
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "SERVER DIED"; tail -20 "$LOG_DIR/server.log" | grep -vE "(Loading|Capturing|Fetching)"; exit 1
fi
echo "server up"

PIDS=()
for ds in $DATASETS; do
  > "$LOG_DIR/${ds}.log"
  PYTHONPATH=$PWD:$PYTHONPATH "$PYTHON" src/craft/run_eval.py \
    --config-name "_eval_tf_${ds}_tmp" > "$LOG_DIR/${ds}.log" 2>&1 &
  PIDS+=($!)
done
echo "evals: ${PIDS[@]}"
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "evals done"
kill $SERVER_PID 2>/dev/null || true
sleep 5
kill -9 $SERVER_PID 2>/dev/null || true

for ds in $DATASETS; do
  LATEST=$(ls -td data/results/${ds}_v1_v1_*/ 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    mkdir -p "$RESULTS_OUT/${ds}"
    cp -r "$LATEST"/. "$RESULTS_OUT/${ds}/"
  fi
done
echo "done ${MODEL_TAG}"
