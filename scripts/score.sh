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
METRIC_ENV_FILE="${METRIC_ENV_FILE:-.uv/metric_envs.env}"

if [ -f "$METRIC_ENV_FILE" ]; then
  # shellcheck source=/dev/null
  source "$METRIC_ENV_FILE"
fi

METRIC_UV_PROJECTS="${METRIC_UV_PROJECTS:-}"
METRIC_UV_PROJECTS_REQUIRED="${METRIC_UV_PROJECTS_REQUIRED:-0}"
METRIC_UV_PROJECT_COMET="${METRIC_UV_PROJECT_COMET:-}"
METRIC_UV_PROJECT_METRICX="${METRIC_UV_PROJECT_METRICX:-}"
METRIC_UV_PROJECT_BLEU="${METRIC_UV_PROJECT_BLEU:-}"

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
    case "$metric" in
      metricx* )
        project="$METRIC_UV_PROJECT_METRICX"
        ;;
      *comet*|xcomet*|cometkiwi* )
        project="$METRIC_UV_PROJECT_COMET"
        ;;
      bleu )
        project="$METRIC_UV_PROJECT_BLEU"
        ;;
    esac
  fi
  if [ -z "$project" ]; then
    if [ "$METRIC_UV_PROJECTS_REQUIRED" = "1" ]; then
      echo "__MISSING__"
      return
    fi
    project="$UV_PROJECT_SCORE"
  fi
  echo "$project"
}

PROJECT=$(metric_project "$METRIC_KEY")
if [ "$PROJECT" = "__MISSING__" ]; then
  echo "ERROR: No UV project configured for metric '$METRIC_KEY'." >&2
  echo "Set METRIC_UV_PROJECTS or METRIC_UV_PROJECT_COMET/METRIC_UV_PROJECT_METRICX/METRIC_UV_PROJECT_BLEU." >&2
  exit 1
fi
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
