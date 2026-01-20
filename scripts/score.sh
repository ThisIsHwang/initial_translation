#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"
METRIC_KEY="${2:?METRIC_KEY required (ex: xcomet_mqm)}"
DATASET="${3:?DATASET required}"
LP="${4:?LP required}"
MODEL_KEY="${5:?MODEL_KEY required}"

uv run evalmt-score \
  --run "$RUN_NAME" \
  --metric "$METRIC_KEY" \
  --dataset "$DATASET" \
  --lp "$LP" \
  --model "$MODEL_KEY"
