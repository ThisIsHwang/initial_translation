#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"
DATASETS="${2:?DATASETS required (comma or all)}"
LPS="${3:-all}"
MODELS="${4:-all}"
METRICS="${5:-all}"
API_BASE="${6:-http://localhost:8000/v1}"

STAGES="${STAGES:-gen,align,score}"

run_stage() {
  local stage="$1"
  case "$stage" in
    gen)
      ./scripts/pipeline_generate.sh "$RUN_NAME" "$DATASETS" "$LPS" "$MODELS" "$API_BASE"
      ;;
    align)
      ./scripts/pipeline_align.sh "$RUN_NAME" "$DATASETS" "$LPS" "$MODELS"
      ;;
    score)
      ./scripts/pipeline_score.sh "$RUN_NAME" "$DATASETS" "$LPS" "$MODELS" "$METRICS"
      ;;
    *)
      echo "Unknown stage: $stage"
      exit 1
      ;;
  esac
}

IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
for STAGE in "${STAGE_LIST[@]}"; do
  run_stage "$STAGE"
done

./scripts/aggregate.sh "$RUN_NAME"
uv run evalmt-aggregate-combos --run "$RUN_NAME"
echo "✅ Done. See outputs/$RUN_NAME/summary.csv and summary_combos.csv"
