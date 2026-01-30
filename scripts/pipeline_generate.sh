#!/usr/bin/env bash
set -euo pipefail

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

if [ "$DOC_GEN_SEP" = "\\n" ]; then
  DOC_GEN_SEP=$'\n'
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for jsonl counting in pipeline_generate.sh"
  exit 1
fi

uv_run_docops() {
  local project="${UV_PROJECT_DOCOPS:-${UV_PROJECT_GEN:-${UV_PROJECT:-}}}"
  if [ -n "$project" ]; then
    uv run --project "$project" "$@"
  else
    uv run "$@"
  fi
}

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

jsonl_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "0"
    return
  fi
  python3 - "$path" <<'PY'
import sys
path = sys.argv[1]
cnt = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cnt += 1
print(cnt)
PY
}

needs_generation() {
  local model_key="$1"
  local need=0
  for dataset in "${DATASET_LIST[@]}"; do
    local doc_dataset="${dataset}${DOC_SUFFIX}"
    read -r -a lp_list <<< "$(get_lp_list "$dataset")"
    for lp in "${lp_list[@]}"; do
      local base_path="data/${dataset}/${lp}.jsonl"
      local doc_path="data/${doc_dataset}/${lp}.jsonl"
      local sent_gen="outputs/${RUN_NAME}/gen/${dataset}/${lp}/${model_key}.jsonl"
      local doc_gen="outputs/${RUN_NAME}/gen/${doc_dataset}/${lp}/${model_key}.jsonl"
      local base_n doc_n sent_n doc_n_gen
      base_n=$(jsonl_count "$base_path")
      doc_n=$(jsonl_count "$doc_path")
      sent_n=$(jsonl_count "$sent_gen")
      doc_n_gen=$(jsonl_count "$doc_gen")
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

resolve_datasets
resolve_models

HOST=$(echo "$API_BASE" | sed -E 's#^https?://([^/:]+).*#\\1#')
PORT=$(echo "$API_BASE" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
if [ -z "$PORT" ] || [ "$PORT" = "$API_BASE" ]; then
  PORT=8000
fi

for dataset in "${DATASET_LIST[@]}"; do
  if [ "$PREPARE_DATA" = "1" ]; then
    ./scripts/prepare_data.sh "$dataset" "$LPS"
  fi
  if [ ! -d "data/$dataset" ]; then
    echo "Missing data dir: data/$dataset (set PREPARE_DATA=1 to build)"
    exit 1
  fi

  read -r -a LP_LIST <<< "$(get_lp_list "$dataset")"
  if [ "${#LP_LIST[@]}" -eq 0 ]; then
    echo "No LPs found for dataset $dataset"
    exit 1
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
    echo "Wrote $DOC_CFG"
  fi

  # Build doc-level datasets
  for lp in "${LP_LIST[@]}"; do
    BASE_PATH="data/${dataset}/${lp}.jsonl"
    if [ ! -f "$BASE_PATH" ]; then
      echo "Missing base dataset: $BASE_PATH"
      exit 1
    fi
    DOC_PATH="${DOC_PREP_DIR}/${lp}.jsonl"
    if [ "$FORCE_DOC_PREP" = "1" ] || [ ! -f "$DOC_PATH" ]; then
      if [ "$DOC_MARKER_ENABLE" = "1" ]; then
        uv_run_docops evalmt-docops to-doc \
          --input "$BASE_PATH" \
          --output "$DOC_PATH" \
          --sep "$DOC_GEN_SEP" \
          --fields "source,reference" \
          --marker-template "$DOC_MARKER_TEMPLATE" \
          --marker-join "$DOC_MARKER_JOIN" \
          --marker-fields "$DOC_MARKER_FIELDS"
      else
        uv_run_docops evalmt-docops to-doc \
          --input "$BASE_PATH" \
          --output "$DOC_PATH" \
          --sep "$DOC_GEN_SEP" \
          --fields "source,reference"
      fi
    fi
  done
done

for MODEL_KEY in "${MODEL_LIST[@]}"; do
  echo "=== Serving model: $MODEL_KEY ==="
  if [ "$MANAGE_SERVER" = "1" ]; then
    ./scripts/stop_vllm.sh "$PORT" || true
  fi

  NEED_GEN=$(needs_generation "$MODEL_KEY")

  SERVER_PID=""
  cleanup() {
    if [ -n "$SERVER_PID" ]; then
      echo "Stopping server PID=$SERVER_PID"
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
      echo "Skipping local serve_vllm (no generation needed)."
    else
      echo "Skipping local serve_vllm (MANAGE_SERVER=$MANAGE_SERVER, host=$HOST)."
    fi
  fi

  if [ "$NEED_GEN" = "1" ]; then
    for dataset in "${DATASET_LIST[@]}"; do
      DOC_DATASET="${dataset}${DOC_SUFFIX}"
      read -r -a LP_LIST <<< "$(get_lp_list "$dataset")"

      for lp in "${LP_LIST[@]}"; do
        ./scripts/generate.sh "$RUN_NAME" "$dataset" "$lp" "$MODEL_KEY" "$API_BASE"
      done

      for lp in "${LP_LIST[@]}"; do
        ./scripts/generate.sh "$RUN_NAME" "$DOC_DATASET" "$lp" "$MODEL_KEY" "$API_BASE"
      done
    done
  else
    echo "All generation outputs already exist for $MODEL_KEY; skipping generation."
  fi

  cleanup
  trap - EXIT || true
done

if [ "$MANAGE_SERVER" = "1" ]; then
  ./scripts/stop_vllm.sh "$PORT" || true
fi
