#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-run1}"
DATASET="wmt24pp"
LPS="${2:-all}"
API_BASE="${3:-http://localhost:8000/v1}"

MODELS="${MODELS:-gpt_oss_120b,qwen3_235b_a22b_instruct_2507,deepseek_v3,gemma3_27b_it,translategemma_27b_it}"
METRICS="${METRICS:-metricx24_qe,xcomet_xl_qe,xcomet_xxl_qe,cometkiwi_wmt23_xxl_qe,cometkiwi_wmt23_xl_qe,cometkiwi_wmt22_qe,cometkiwi_wmt23_xxl_qe_ctx,cometkiwi_wmt23_xl_qe_ctx,cometkiwi_wmt22_qe_ctx}"

# Defaults for full automation (override as needed)
MANAGE_SERVER="${MANAGE_SERVER:-1}"
DOC_ALIGN_MODE="${DOC_ALIGN_MODE:-gpt}"
MANAGE_ALIGN_SERVER="${MANAGE_ALIGN_SERVER:-1}"
DOC_ALIGN_API_BASE="${DOC_ALIGN_API_BASE:-http://localhost:8001/v1}"
DOC_ALIGN_MODEL_KEY="${DOC_ALIGN_MODEL_KEY:-gpt_oss_120b}"
DOC_ALIGN_MODEL_NAME="${DOC_ALIGN_MODEL_NAME:-gpt-oss-120b}"
DOC_ALIGN_RESPONSE_FORMAT="${DOC_ALIGN_RESPONSE_FORMAT:-json_schema}"

if [ "$DOC_ALIGN_MODE" = "gpt" ]; then
  DOC_ALIGN_MODEL_KEY="gpt_oss_120b"
  DOC_ALIGN_MODEL_NAME="gpt-oss-120b"
fi

MANAGE_SERVER="$MANAGE_SERVER" \
DOC_ALIGN_MODE="$DOC_ALIGN_MODE" \
MANAGE_ALIGN_SERVER="$MANAGE_ALIGN_SERVER" \
DOC_ALIGN_API_BASE="$DOC_ALIGN_API_BASE" \
DOC_ALIGN_MODEL_KEY="$DOC_ALIGN_MODEL_KEY" \
DOC_ALIGN_MODEL_NAME="$DOC_ALIGN_MODEL_NAME" \
DOC_ALIGN_RESPONSE_FORMAT="$DOC_ALIGN_RESPONSE_FORMAT" \
bash scripts/pipeline_all.sh "$RUN_NAME" "$DATASET" "$LPS" "$MODELS" "$METRICS" "$API_BASE"
