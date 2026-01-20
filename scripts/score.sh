#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"
METRIC_KEY="${2:?METRIC_KEY required (ex: xcomet_mqm)}"
DATASET="${3:?DATASET required}"
LP="${4:?LP required}"
MODEL_KEY="${5:?MODEL_KEY required}"

# Optional: restrict GPUs for scoring (useful if you keep a vLLM server running
# on other GPUs and want to score on a free GPU).
# Example:
#   SCORE_GPU_LIST=7 ./scripts/score.sh ...
SCORE_GPU_LIST="${SCORE_GPU_LIST:-}"

if [ -n "$SCORE_GPU_LIST" ]; then
  echo "CUDA_VISIBLE_DEVICES=$SCORE_GPU_LIST"
  CUDA_VISIBLE_DEVICES="$SCORE_GPU_LIST" uv run evalmt-score \
    --run "$RUN_NAME" \
    --metric "$METRIC_KEY" \
    --dataset "$DATASET" \
    --lp "$LP" \
    --model "$MODEL_KEY"
else
  uv run evalmt-score \
    --run "$RUN_NAME" \
    --metric "$METRIC_KEY" \
    --dataset "$DATASET" \
    --lp "$LP" \
    --model "$MODEL_KEY"
fi
