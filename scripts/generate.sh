#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required (ex: run1)}"
DATASET="${2:?DATASET required (ex: wmt24pp)}"
LP="${3:?LP required (ex: en-ko_KR)}"
MODEL_KEY="${4:?MODEL_KEY required (ex: gpt_oss_120b)}"
API_BASE="${5:-http://localhost:8000/v1}"

# You can override concurrency like:
#   CONCURRENCY=64 ./scripts/generate.sh ...
CONCURRENCY="${CONCURRENCY:-16}"

uv run evalmt-generate \
  --run "$RUN_NAME" \
  --dataset "$DATASET" \
  --lp "$LP" \
  --model "$MODEL_KEY" \
  --api-base "$API_BASE" \
  --concurrency "$CONCURRENCY" \
  --resume
