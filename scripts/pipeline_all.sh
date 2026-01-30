#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=./_pipeline_lib.sh
source "$SCRIPT_DIR/_pipeline_lib.sh"

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
      pipeline_die "Unknown stage: $stage"
      ;;
  esac
}

IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
for STAGE in "${STAGE_LIST[@]}"; do
  pipeline_log "== Stage: $STAGE =="
  run_stage "$STAGE"
done

./scripts/aggregate.sh "$RUN_NAME"
uv run evalmt-aggregate-combos --run "$RUN_NAME"
pipeline_log "✅ Done. See outputs/$RUN_NAME/summary.csv and summary_combos.csv"
