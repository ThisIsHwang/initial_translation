#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"
DATASETS="${2:?DATASETS required (comma or all)}"
LPS="${3:-all}"
MODELS="${4:-all}"
METRICS="${5:-all}"

DOC_SUFFIX="${DOC_SUFFIX:-_doc}"

resolve_datasets() {
  if [ "$DATASETS" = "all" ]; then
    DATASET_LIST=()
    if [ -d configs/datasets ]; then
      mapfile -t DATASET_LIST < <(ls configs/datasets/*.yaml 2>/dev/null | sed 's#.*/##; s/\\.yaml$//' | rg -v '_doc$' || true)
    fi
    if [ "${#DATASET_LIST[@]}" -eq 0 ] && [ -d data ]; then
      mapfile -t DATASET_LIST < <(ls -1 data 2>/dev/null | rg -v '_doc$' || true)
    fi
  else
    IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
  fi
  if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
    echo "No datasets found."
    exit 1
  fi
}

resolve_models() {
  if [ "$MODELS" = "all" ]; then
    mapfile -t MODEL_LIST < <(ls configs/models/*.yaml 2>/dev/null | sed 's#.*/##; s/\\.yaml$//' || true)
  else
    IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
  fi
  if [ "${#MODEL_LIST[@]}" -eq 0 ]; then
    echo "No models found."
    exit 1
  fi
}

resolve_metrics() {
  if [ "$METRICS" = "all" ]; then
    mapfile -t METRIC_LIST < <(ls configs/metrics/*.yaml 2>/dev/null | sed 's#.*/##; s/\\.yaml$//' || true)
  else
    IFS=',' read -r -a METRIC_LIST <<< "$METRICS"
  fi
  if [ "${#METRIC_LIST[@]}" -eq 0 ]; then
    echo "No metrics found."
    exit 1
  fi
}

get_lp_list() {
  local dataset="$1"
  if [ "$LPS" = "all" ]; then
    if [ -d "data/$dataset" ]; then
      ls "data/$dataset"/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\\.jsonl$//' | sort -u
    fi
  else
    echo "${LPS//,/ }"
  fi
}

dataset_has_reference() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "0"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$path" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            print(0)
            sys.exit(0)
        print(1 if "reference" in row else 0)
        sys.exit(0)
print(0)
PY
    return
  fi
  if rg -q '"reference"' "$path"; then
    echo "1"
  else
    echo "0"
  fi
}

metric_requires_reference() {
  local metric="$1"
  local cfg="configs/metrics/${metric}.yaml"
  if [ ! -f "$cfg" ]; then
    return 1
  fi
  if rg -q '^mode:\\s*ref' "$cfg"; then
    return 0
  fi
  if rg -q 'requires_reference:\\s*true' "$cfg"; then
    return 0
  fi
  return 1
}

resolve_datasets
resolve_models
resolve_metrics

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
  read -r -a LP_LIST <<< "$(get_lp_list "$dataset")"
  if [ "${#LP_LIST[@]}" -eq 0 ]; then
    echo "No LPs found for dataset $dataset"
    continue
  fi

  DOC_DATASET="${dataset}${DOC_SUFFIX}"

  for MODEL_KEY in "${MODEL_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      BASE_PATH="data/${dataset}/${LP}.jsonl"
      DOC_PATH="data/${DOC_DATASET}/${LP}.jsonl"
      HAS_REF=$(dataset_has_reference "$BASE_PATH")

      # sentence evals (non-context)
      for METRIC in "${METRICS_SENT[@]}"; do
        if [ "$HAS_REF" = "0" ] && metric_requires_reference "$METRIC"; then
          echo "Skip metric $METRIC (no reference in $BASE_PATH)"
          continue
        fi
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "$MODEL_KEY"
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "${MODEL_KEY}__from_doc"
      done

      # document evals (context)
      for METRIC in "${METRICS_DOC[@]}"; do
        if [ "$HAS_REF" = "0" ] && metric_requires_reference "$METRIC"; then
          echo "Skip metric $METRIC (no reference in $BASE_PATH)"
          continue
        fi
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "$MODEL_KEY"
        ./scripts/score.sh "$RUN_NAME" "$METRIC" "$dataset" "$LP" "${MODEL_KEY}__from_doc"
      done

      # document evals (non-context) for doc input -> doc eval
      if [ -f "$DOC_PATH" ]; then
        DOC_HAS_REF=$(dataset_has_reference "$DOC_PATH")
        for METRIC in "${METRICS_SENT[@]}"; do
          if [ "$DOC_HAS_REF" = "0" ] && metric_requires_reference "$METRIC"; then
            echo "Skip metric $METRIC (no reference in $DOC_PATH)"
            continue
          fi
          ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DOC_DATASET" "$LP" "$MODEL_KEY"
        done
      fi
    done
  done
done
