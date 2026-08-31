#!/bin/bash
# Run all 4 training-format evals (3 in-dist + 1 OOD) sequentially against one server.
set -e
PYTHON="${PYTHON:-python}"
MODEL_PATH="$1"
MODEL_TAG="$2"
RESULTS_OUT="data/results/tf/${MODEL_TAG}"
mkdir -p "$RESULTS_OUT"
LOG_DIR="/tmp/tf-${MODEL_TAG}"
mkdir -p "$LOG_DIR"

echo "==== Booting server for ${MODEL_TAG} ===="
> "$LOG_DIR/server.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_TRITON_FLASH_ATTN=0 \
nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --port 8002 --host 0.0.0.0 --dtype bfloat16 \
  --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "server PID=$SERVER_PID"

until curl -sf http://localhost:8002/v1/models > /dev/null 2>&1 || ! kill -0 $SERVER_PID 2>/dev/null; do sleep 8; done
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "SERVER DIED. tail:"; tail -30 "$LOG_DIR/server.log" | grep -vE "(Loading|Capturing|Fetching)"; exit 1
fi
echo "server up"

# Run all 4 datasets in parallel — vLLM TP=4 batches all together
PIDS=()
for ds in hotpotqa musique 2wiki multihoprag; do
  > "$LOG_DIR/${ds}.log"
  PYTHONPATH=$PWD:$PYTHONPATH "$PYTHON" src/craft/run_eval.py \
    --config-name "_eval_tf_${ds}_tmp" > "$LOG_DIR/${ds}.log" 2>&1 &
  PIDS+=($!)
done
echo "evals started: ${PIDS[@]}"
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "evals done"

# Stop server
kill $SERVER_PID 2>/dev/null || true
sleep 5
kill -9 $SERVER_PID 2>/dev/null || true

# Collate results into named output
for ds in hotpotqa musique 2wiki multihoprag; do
  LATEST=$(ls -td data/results/${ds}_v1_v1_*/ 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    mkdir -p "$RESULTS_OUT/${ds}"
    cp -r "$LATEST"/. "$RESULTS_OUT/${ds}/"
  fi
done

echo ""
echo "==== ${MODEL_TAG} summaries ===="
for ds in hotpotqa musique 2wiki multihoprag; do
  s="$RESULTS_OUT/$ds/summary.json"
  [ -f "$s" ] && "$PYTHON" -c "
import json
d = json.load(open('$s'))
print(f'  {\"$MODEL_TAG\":20s} {\"$ds\":12s}  EM={d[\"em_score\"]*100:5.2f} F1={d[\"f1_score\"]*100:5.2f} Fmt={d[\"format_score\"]*100:5.2f} Acc={d[\"accuracy_score\"]*100:5.2f} Rel={d[\"relevance_score\"]*100:5.2f}')
"
done
