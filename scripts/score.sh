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
UV_PROJECT_SCORE="${UV_PROJECT_SCORE:-${UV_PROJECT:-}}"
METRIC_UV_PROJECTS="${METRIC_UV_PROJECTS:-}"

metric_project() {
  local metric="$1"
  local project=""
  if [ -n "$METRIC_UV_PROJECTS" ]; then
    IFS=',' read -r -a pairs <<< "$METRIC_UV_PROJECTS"
    for pair in "${pairs[@]}"; do
      local key="${pair%%=*}"
      local val="${pair#*=}"
      if [ "$key" = "$metric" ]; then
        project="$val"
        break
      fi
    done
  fi
  if [ -z "$project" ]; then
    project="$UV_PROJECT_SCORE"
  fi
  echo "$project"
}

PROJECT=$(metric_project "$METRIC_KEY")
UV_ARGS=()
if [ -n "$PROJECT" ]; then
  UV_ARGS=(--project "$PROJECT")
fi

if [ -n "$SCORE_GPU_LIST" ]; then
  echo "CUDA_VISIBLE_DEVICES=$SCORE_GPU_LIST"
  CUDA_VISIBLE_DEVICES="$SCORE_GPU_LIST" uv run "${UV_ARGS[@]}" evalmt-score \
    --run "$RUN_NAME" \
    --metric "$METRIC_KEY" \
    --dataset "$DATASET" \
    --lp "$LP" \
    --model "$MODEL_KEY"
else
  uv run "${UV_ARGS[@]}" evalmt-score \
    --run "$RUN_NAME" \
    --metric "$METRIC_KEY" \
    --dataset "$DATASET" \
    --lp "$LP" \
    --model "$MODEL_KEY"
fi
