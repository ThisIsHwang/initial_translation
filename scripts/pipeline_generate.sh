#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=./_pipeline_lib.sh
source "$SCRIPT_DIR/_pipeline_lib.sh"

RUN_NAME="${1:?RUN_NAME required}"
DATASETS="${2:?DATASETS required (comma or all)}"
LPS="${3:-all}"
MODELS="${4:-all}"
API_BASE="${5:-http://localhost:8000/v1}"

MANAGE_SERVER="${MANAGE_SERVER:-0}"
PREPARE_DATA="${PREPARE_DATA:-0}"
FORCE_DOC_PREP="${FORCE_DOC_PREP:-0}"

DOC_SUFFIX="${DOC_SUFFIX:-_doc}"
DOC_GEN_SEP="${DOC_GEN_SEP:-$'\n'}"
DOC_MARKER_ENABLE="${DOC_MARKER_ENABLE:-1}"
DOC_MARKER_TEMPLATE="${DOC_MARKER_TEMPLATE:-⟦{i}⟧}"
DOC_MARKER_JOIN="${DOC_MARKER_JOIN:- }"
DOC_MARKER_FIELDS="${DOC_MARKER_FIELDS:-source}"

DOC_GEN_SEP="$(pipeline_normalize_sep "$DOC_GEN_SEP")"

pipeline_require_cmd python3

mapfile -t DATASET_LIST < <(pipeline_list_datasets "$DATASETS")
[ "${#DATASET_LIST[@]}" -gt 0 ] || pipeline_die "No datasets found."
mapfile -t MODEL_LIST < <(pipeline_list_models "$MODELS")
[ "${#MODEL_LIST[@]}" -gt 0 ] || pipeline_die "No models found."

read -r HOST PORT < <(pipeline_api_host_port "$API_BASE")

needs_generation() {
  local model_key="$1"
  local need=0
  for dataset in "${DATASET_LIST[@]}"; do
    local base_dir
    base_dir="$(pipeline_dataset_prepared_dir "$dataset")"
    local doc_dataset="${dataset}${DOC_SUFFIX}"
    local doc_dir
    doc_dir="$(pipeline_dataset_prepared_dir "$doc_dataset")"
    mapfile -t lp_list < <(pipeline_list_lps "$dataset" "$LPS")
    for lp in "${lp_list[@]}"; do
      local base_path="${base_dir}/${lp}.jsonl"
      local doc_path="${doc_dir}/${lp}.jsonl"
      local sent_gen="outputs/${RUN_NAME}/gen/${dataset}/${lp}/${model_key}.jsonl"
      local doc_gen="outputs/${RUN_NAME}/gen/${doc_dataset}/${lp}/${model_key}.jsonl"
      local base_n doc_n sent_n doc_n_gen
      base_n=$(pipeline_jsonl_count "$base_path")
      doc_n=$(pipeline_jsonl_count "$doc_path")
      sent_n=$(pipeline_jsonl_count "$sent_gen")
      doc_n_gen=$(pipeline_jsonl_count "$doc_gen")
      if [ "$sent_n" -lt "$base_n" ] || [ "$doc_n_gen" -lt "$doc_n" ]; then
        need=1
        break
      fi
    done
    if [ "$need" = "1" ]; then
      break
    fi
  done
  echo "$need"
}

for dataset in "${DATASET_LIST[@]}"; do
  BASE_DIR="$(pipeline_dataset_prepared_dir "$dataset")"
  if [ "$PREPARE_DATA" = "1" ]; then
    ./scripts/prepare_data.sh "$dataset" "$LPS"
  fi
  if [ ! -d "$BASE_DIR" ]; then
    pipeline_die "Missing data dir: $BASE_DIR (set PREPARE_DATA=1 to build)"
  fi

  mapfile -t LP_LIST < <(pipeline_list_lps "$dataset" "$LPS")
  if [ "${#LP_LIST[@]}" -eq 0 ]; then
    pipeline_die "No LPs found for dataset $dataset"
  fi

  # Ensure doc dataset config
  DOC_DATASET="${dataset}${DOC_SUFFIX}"
  DOC_PREP_DIR="data/${DOC_DATASET}"
  DOC_CFG="configs/datasets/${DOC_DATASET}.yaml"
  mkdir -p "$DOC_PREP_DIR"
  if [ ! -f "$DOC_CFG" ]; then
    cat > "$DOC_CFG" <<EOF
type: ${dataset}
prepared_dir: ${DOC_PREP_DIR}
EOF
    pipeline_log "Wrote $DOC_CFG"
  fi

  # Build doc-level datasets
  for lp in "${LP_LIST[@]}"; do
    BASE_PATH="${BASE_DIR}/${lp}.jsonl"
    if [ ! -f "$BASE_PATH" ]; then
      echo "Missing base dataset: $BASE_PATH"
      exit 1
    fi
    DOC_PATH="${DOC_PREP_DIR}/${lp}.jsonl"
    if [ "$FORCE_DOC_PREP" = "1" ] || [ ! -f "$DOC_PATH" ]; then
      if [ "$DOC_MARKER_ENABLE" = "1" ]; then
        pipeline_docops to-doc \
          --input "$BASE_PATH" \
          --output "$DOC_PATH" \
          --sep "$DOC_GEN_SEP" \
          --fields "source,reference" \
          --marker-template "$DOC_MARKER_TEMPLATE" \
          --marker-join "$DOC_MARKER_JOIN" \
          --marker-fields "$DOC_MARKER_FIELDS"
      else
        pipeline_docops to-doc \
          --input "$BASE_PATH" \
          --output "$DOC_PATH" \
          --sep "$DOC_GEN_SEP" \
          --fields "source,reference"
      fi
    fi
  done
done

for MODEL_KEY in "${MODEL_LIST[@]}"; do
  pipeline_log "=== Serving model: $MODEL_KEY ==="
  if [ "$MANAGE_SERVER" = "1" ]; then
    ./scripts/stop_vllm.sh "$PORT" || true
  fi

  NEED_GEN=$(needs_generation "$MODEL_KEY")

  SERVER_PID=""
  cleanup() {
    if [ -n "$SERVER_PID" ]; then
      pipeline_log "Stopping server PID=$SERVER_PID"
      kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
      kill -TERM "$SERVER_PID" 2>/dev/null || true
      sleep 2
      if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        kill -KILL "$SERVER_PID" 2>/dev/null || true
      fi
    fi
  }

  if [ "$NEED_GEN" = "1" ] && [ "$MANAGE_SERVER" = "1" ] && { [ "$HOST" = "localhost" ] || [ "$HOST" = "127.0.0.1" ]; }; then
    if command -v setsid >/dev/null 2>&1; then
      (setsid ./scripts/serve_vllm.sh "$MODEL_KEY" "$PORT") &
    else
      (./scripts/serve_vllm.sh "$MODEL_KEY" "$PORT") &
    fi
    SERVER_PID=$!
    trap cleanup EXIT
    ./scripts/wait_server.sh "$API_BASE" 600
  else
    if [ "$NEED_GEN" = "0" ]; then
      pipeline_log "Skipping local serve_vllm (no generation needed)."
    else
      pipeline_log "Skipping local serve_vllm (MANAGE_SERVER=$MANAGE_SERVER, host=$HOST)."
    fi
  fi

  if [ "$NEED_GEN" = "1" ]; then
    for dataset in "${DATASET_LIST[@]}"; do
      DOC_DATASET="${dataset}${DOC_SUFFIX}"
      mapfile -t LP_LIST < <(pipeline_list_lps "$dataset" "$LPS")

      for lp in "${LP_LIST[@]}"; do
        ./scripts/generate.sh "$RUN_NAME" "$dataset" "$lp" "$MODEL_KEY" "$API_BASE"
      done

      for lp in "${LP_LIST[@]}"; do
        ./scripts/generate.sh "$RUN_NAME" "$DOC_DATASET" "$lp" "$MODEL_KEY" "$API_BASE"
      done
    done
  else
    pipeline_log "All generation outputs already exist for $MODEL_KEY; skipping generation."
  fi

  cleanup
  trap - EXIT || true
done

if [ "$MANAGE_SERVER" = "1" ]; then
  ./scripts/stop_vllm.sh "$PORT" || true
fi
