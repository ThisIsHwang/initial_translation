#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required (ex: run1)}"
DATASET="${2:?DATASET required (ex: wmt24pp)}"
LP="${3:?LP required (ex: en-ko_KR)}"
MODEL_KEY="${4:?MODEL_KEY required (ex: gemma3_27b_it)}"
API_BASE="${5:-http://localhost:8000/v1}"

# You can override concurrency like:
#   CONCURRENCY=64 ./scripts/generate.sh ...
CONCURRENCY="${CONCURRENCY:-16}"
UV_PROJECT_GEN="${UV_PROJECT_GEN:-${UV_PROJECT:-}}"

UV_ARGS=()
if [ -n "$UV_PROJECT_GEN" ]; then
  UV_ARGS=(--project "$UV_PROJECT_GEN")
fi

uv run "${UV_ARGS[@]}" evalmt-generate \
  --run "$RUN_NAME" \
  --dataset "$DATASET" \
  --lp "$LP" \
  --model "$MODEL_KEY" \
  --api-base "$API_BASE" \
  --concurrency "$CONCURRENCY" \
  --resume
