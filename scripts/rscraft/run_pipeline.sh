#!/bin/bash
# ========================================
# RS-CRAFT pipeline orchestrator
#
# Phases:
#   A  sample N candidate traces from base 7B (vLLM)         -> candidates.jsonl
#   B  score with R_fmt+R_gold+R_ans+R_faith (Qwen3-30B local) -> scored.jsonl
#   C  select best-of-N + filter                              -> rscraft_best_of_N_v1.jsonl
#   D  SFT Qwen2.5-7B on the filtered set                     -> checkpoints/RS-CRAFT-7B-v1
#   E  evaluate on hotpotqa / musique / 2wiki
#
# This script does NOT auto-launch GPU servers — the user controls server
# placement. It prints the commands to run for each phase and which one to
# execute next based on which output files exist.
# ========================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PHASE=${1:-help}
LIMIT=${LIMIT:-}             # optional pilot limit, e.g. LIMIT=500
CONFIG=${CONFIG:-rscraft}    # cfg/<CONFIG>.yaml

CANDIDATES=$(python3 -c "import yaml; c=yaml.safe_load(open('cfg/${CONFIG}.yaml')); print(c['candidates_path'])")
SCORED=$(python3 -c "import yaml; c=yaml.safe_load(open('cfg/${CONFIG}.yaml')); print(c['scored_path'])")
SELECTED=$(python3 -c "import yaml; c=yaml.safe_load(open('cfg/${CONFIG}.yaml')); print(c['selected_path'])")

echo "=================================================="
echo "RS-CRAFT pipeline (phase = $PHASE)"
echo "  candidates : $CANDIDATES   $([ -f $CANDIDATES ] && echo '[present]' || echo '[missing]')"
echo "  scored     : $SCORED       $([ -f $SCORED ] && echo '[present]' || echo '[missing]')"
echo "  selected   : $SELECTED     $([ -f $SELECTED ] && echo '[present]' || echo '[missing]')"
echo "=================================================="

case $PHASE in
  A|sample)
    echo "Phase A: sampling N candidates (uses base 7B vLLM)."
    echo "Pre-req: scripts/server/start_craft.sh on http://localhost:8002"
    LIMIT_ARG=""
    [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
    python scripts/rscraft/sample_candidates.py --config "$CONFIG" $LIMIT_ARG
    ;;

  B|score)
    echo "Phase B: scoring candidates with R_fmt+R_gold+R_ans+R_faith."
    echo "Pre-req: scripts/server/start_judge.sh on http://localhost:8000 (Qwen3-30B)"
    LIMIT_ARG=""
    [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
    python scripts/rscraft/score_candidates.py --config "$CONFIG" $LIMIT_ARG
    ;;

  C|select)
    echo "Phase C: selecting best-of-N (CPU only)."
    python scripts/rscraft/select_best.py --config "$CONFIG"
    ;;

  D|sft)
    echo "Phase D: SFT on filtered traces."
    echo "Make sure no vLLM server occupies your GPUs first (kill judge & base servers)."
    NUM_GPUS=${NUM_GPUS:-4} CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3} \
      ./scripts/train/sft.sh --config sft_rscraft
    ;;

  E|eval)
    echo "Phase E: eval on three datasets."
    echo "Pre-req: scripts/server/start_craft.sh pointing at checkpoints/RS-CRAFT-7B-v1"
    for ds in hotpotqa musique 2wiki; do
      ./scripts/eval/eval.sh --version v1 --dataset $ds
    done
    ;;

  pilot)
    echo "Pilot run on LIMIT=${LIMIT:-500} prompts (sampling+scoring+select)."
    LIMIT=${LIMIT:-500}
    LIMIT=$LIMIT bash $0 A
    LIMIT=$LIMIT bash $0 B
    bash $0 C
    ;;

  help|*)
    cat <<EOF

Usage:  scripts/rscraft/run_pipeline.sh <phase>

Phases:
  A | sample        Stage A — sample N candidates from base 7B vLLM
  B | score         Stage B — score with the four-reward auditor
  C | select        Stage C — best-of-N + filter -> SFT JSONL
  D | sft           Stage D — full-parameter SFT (cfg/sft_rscraft.yaml)
  E | eval          Stage E — eval on hotpotqa/musique/2wiki
  pilot             A+B+C with LIMIT=500 (default) for end-to-end smoke

Env overrides:
  LIMIT=N            cap prompts (pilot mode)
  CONFIG=<name>      use cfg/<name>.yaml instead of cfg/rscraft.yaml
  NUM_GPUS, CUDA_DEVICES   passed through to SFT phase

GPU choreography (4xH100):
  Phases A:        base 7B   on GPU 0,1,2,3 (TP=4)  - kill server when done
  Phase  B:        Qwen3-30B on GPU 0,1   (TP=2)    - judge can stay running
  Phase  C:        CPU only
  Phase  D:        kill judge first; SFT on GPU 0,1,2,3
  Phase  E:        RS-CRAFT-7B-v1 on GPU 0,1,2,3 (TP=4)

EOF
    ;;
esac
