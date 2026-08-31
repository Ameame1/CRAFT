#!/bin/bash
# Boot vLLM TP=4 server for given model, run both OOD evals in parallel, kill server.
set -e
PYTHON="${PYTHON:-python}"
MODEL_PATH="$1"
MODEL_TAG="$2"  # short name for logs
RESULTS_OUT="data/results/ood/${MODEL_TAG}"
mkdir -p "$RESULTS_OUT"
LOG_DIR="/tmp/ood-${MODEL_TAG}"
mkdir -p "$LOG_DIR"

echo "==== Booting server for ${MODEL_TAG} ===="
> "$LOG_DIR/server.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_USE_TRITON_FLASH_ATTN=0 \
nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --port 8002 --host 0.0.0.0 --dtype bfloat16 \
  --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "server PID=$SERVER_PID"

# wait for server
echo "waiting for /v1/models..."
until curl -sf http://localhost:8002/v1/models > /dev/null 2>&1 || ! kill -0 $SERVER_PID 2>/dev/null; do
  sleep 8
done
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "SERVER DIED. tail:"
  tail -30 "$LOG_DIR/server.log" | grep -vE "(Loading checkpoint|Capturing CUDA|Fetching)"
  exit 1
fi
echo "server up"

# Run both evals in parallel
> "$LOG_DIR/multihoprag.log"
> "$LOG_DIR/fanoutqa.log"
PYTHONPATH=$PWD:$PYTHONPATH "$PYTHON" src/craft/run_eval.py \
  --config-name "_eval_ood_multihoprag_tmp" > "$LOG_DIR/multihoprag.log" 2>&1 &
P1=$!
PYTHONPATH=$PWD:$PYTHONPATH "$PYTHON" src/craft/run_eval.py \
  --config-name "_eval_ood_fanoutqa_tmp" > "$LOG_DIR/fanoutqa.log" 2>&1 &
P2=$!
echo "evals: multihoprag=$P1 fanoutqa=$P2"
wait $P1; ec1=$?
wait $P2; ec2=$?
echo "evals done: mhr exit=$ec1, foq exit=$ec2"

# Stop server
kill $SERVER_PID 2>/dev/null || true
sleep 5
kill -9 $SERVER_PID 2>/dev/null || true

# Move latest result dirs into named output
LATEST_MHR=$(ls -td data/results/multihoprag_v1_v1_*/ 2>/dev/null | head -1)
LATEST_FOQ=$(ls -td data/results/fanoutqa_v1_v1_*/ 2>/dev/null | head -1)
[ -n "$LATEST_MHR" ] && mkdir -p "$RESULTS_OUT/multihoprag" && cp -r "$LATEST_MHR"/. "$RESULTS_OUT/multihoprag/"
[ -n "$LATEST_FOQ" ] && mkdir -p "$RESULTS_OUT/fanoutqa" && cp -r "$LATEST_FOQ"/. "$RESULTS_OUT/fanoutqa/"

echo ""
echo "==== ${MODEL_TAG} summary ===="
[ -f "$RESULTS_OUT/multihoprag/summary.json" ] && cat "$RESULTS_OUT/multihoprag/summary.json"
echo ""
[ -f "$RESULTS_OUT/fanoutqa/summary.json" ] && cat "$RESULTS_OUT/fanoutqa/summary.json"
