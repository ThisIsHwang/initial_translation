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

DOC_SUFFIX="${DOC_SUFFIX:-_doc}"
METRIC_ENV_FILE="${METRIC_ENV_FILE:-.uv/metric_envs.env}"

if [ -f "$METRIC_ENV_FILE" ]; then
  # shellcheck source=/dev/null
  source "$METRIC_ENV_FILE"
fi

mapfile -t DATASET_LIST < <(pipeline_list_datasets "$DATASETS")
[ "${#DATASET_LIST[@]}" -gt 0 ] || pipeline_die "No datasets found."
mapfile -t MODEL_LIST < <(pipeline_list_models "$MODELS")
[ "${#MODEL_LIST[@]}" -gt 0 ] || pipeline_die "No models found."
mapfile -t METRIC_LIST < <(pipeline_list_metrics "$METRICS")
[ "${#METRIC_LIST[@]}" -gt 0 ] || pipeline_die "No metrics found."

# Split metrics: sentence-eval uses non-context; doc-eval uses context (if provided).
METRICS_SENT=()
METRICS_DOC=()
if [ -n "${METRICS_SENT_OVERRIDE:-}" ]; then
  IFS=',' read -r -a METRICS_SENT <<< "$METRICS_SENT_OVERRIDE"
fi
if [ -n "${METRICS_DOC_OVERRIDE:-}" ]; then
  IFS=',' read -r -a METRICS_DOC <<< "$METRICS_DOC_OVERRIDE"
fi

if [ "${#METRICS_SENT[@]}" -eq 0 ] || [ "${#METRICS_DOC[@]}" -eq 0 ]; then
  for M in "${METRIC_LIST[@]}"; do
    if [[ "$M" == *_ctx ]]; then
      METRICS_DOC+=("$M")
    else
      METRICS_SENT+=("$M")
    fi
  done
fi

if [ "${#METRICS_DOC[@]}" -eq 0 ]; then
  echo "No context metrics provided; falling back to non-context for doc eval."
  METRICS_DOC=("${METRICS_SENT[@]}")
fi

for dataset in "${DATASET_LIST[@]}"; do
  mapfile -t LP_LIST < <(pipeline_list_lps "$dataset" "$LPS")
  if [ "${#LP_LIST[@]}" -eq 0 ]; then
    pipeline_log "No LPs found for dataset $dataset"
    continue
  fi

  DOC_DATASET="${dataset}${DOC_SUFFIX}"

  for MODEL_KEY in "${MODEL_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      BASE_PATH="data/${dataset}/${LP}.jsonl"
      DOC_PATH="data/${DOC_DATASET}/${LP}.jsonl"
      HAS_REF=$(pipeline_jsonl_has_key "$BASE_PATH" "reference")

      # sentence evals (non-context)
      for METRIC in "${METRICS_SENT[@]}"; do
        if [ "$HAS_REF" = "0" ] && pipeline_metric_requires_reference "$METRIC"; then
          pipeline_log "Skip metric $METRIC (no reference in $BASE_PATH)"
          continue
        fi
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "$MODEL_KEY"
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "${MODEL_KEY}__from_doc"
      done

      # document evals (context)
      for METRIC in "${METRICS_DOC[@]}"; do
        if [ "$HAS_REF" = "0" ] && pipeline_metric_requires_reference "$METRIC"; then
          pipeline_log "Skip metric $METRIC (no reference in $BASE_PATH)"
          continue
        fi
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "$MODEL_KEY"
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "${MODEL_KEY}__from_doc"
      done

      # document evals (non-context) for doc input -> doc eval
      if [ -f "$DOC_PATH" ]; then
        DOC_HAS_REF=$(pipeline_jsonl_has_key "$DOC_PATH" "reference")
        for METRIC in "${METRICS_SENT[@]}"; do
          if [ "$DOC_HAS_REF" = "0" ] && pipeline_metric_requires_reference "$METRIC"; then
            pipeline_log "Skip metric $METRIC (no reference in $DOC_PATH)"
            continue
          fi
          ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DOC_DATASET" "$LP" "$MODEL_KEY"
        done
      fi
    done
  done
done
