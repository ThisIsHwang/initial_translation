#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=./_pipeline_lib.sh
source "$SCRIPT_DIR/_pipeline_lib.sh"

RUN_NAME="${1:?RUN_NAME required}"
DATASETS="${2:?DATASETS required (comma or all)}"
LPS="${3:-all}"
MODELS="${4:-all}"

DOC_SUFFIX="${DOC_SUFFIX:-_doc}"
DOC_GEN_SEP="${DOC_GEN_SEP:-$'\n'}"
DOC_SPLIT_SEP="${DOC_SPLIT_SEP:-$DOC_GEN_SEP}"
DOC_MARKER_ENABLE="${DOC_MARKER_ENABLE:-1}"
DOC_MARKER_TEMPLATE="${DOC_MARKER_TEMPLATE:-⟦{i}⟧}"
DOC_MARKER_JOIN="${DOC_MARKER_JOIN:- }"
DOC_MARKER_FIELDS="${DOC_MARKER_FIELDS:-source}"
DOC_MARKER_REGEX="${DOC_MARKER_REGEX:-⟦\\d+⟧}"
DOC_MARKER_KEEP_RAW="${DOC_MARKER_KEEP_RAW:-1}"

DOC_ALIGN_MODE="${DOC_ALIGN_MODE:-rule}"
DOC_ALIGN_META="${DOC_ALIGN_META:-0}"
DOC_ALIGN_MODEL="${DOC_ALIGN_MODEL:-}"
DOC_ALIGN_API_BASE="${DOC_ALIGN_API_BASE:-http://localhost:8001/v1}"
DOC_ALIGN_MODEL_KEY="${DOC_ALIGN_MODEL_KEY:-gpt_oss_120b}"
DOC_ALIGN_MODEL_NAME="${DOC_ALIGN_MODEL_NAME:-gpt-oss-120b}"
DOC_ALIGN_MAX_TOKENS="${DOC_ALIGN_MAX_TOKENS:-64000}"
DOC_ALIGN_RESPONSE_FORMAT="${DOC_ALIGN_RESPONSE_FORMAT:-json_schema}"
MANAGE_ALIGN_SERVER="${MANAGE_ALIGN_SERVER:-0}"

CLEAN_GPU="${CLEAN_GPU:-1}"

DOC_GEN_SEP="$(pipeline_normalize_sep "$DOC_GEN_SEP")"
DOC_SPLIT_SEP="$(pipeline_normalize_sep "$DOC_SPLIT_SEP")"

mapfile -t DATASET_LIST < <(pipeline_list_datasets "$DATASETS")
[ "${#DATASET_LIST[@]}" -gt 0 ] || pipeline_die "No datasets found."
mapfile -t MODEL_LIST < <(pipeline_list_models "$MODELS")
[ "${#MODEL_LIST[@]}" -gt 0 ] || pipeline_die "No models found."

if [ "$CLEAN_GPU" = "1" ]; then
  ./scripts/clean_gpu.sh
fi

if [ "$DOC_ALIGN_MODE" = "gpt" ]; then
  DOC_ALIGN_MODEL_KEY="gpt_oss_120b"
  DOC_ALIGN_MODEL_NAME="gpt-oss-120b"
  pipeline_log "Align model forced to gpt-oss (gpt_oss_120b / gpt-oss-120b)."
fi

ALIGN_PID=""
if [ "$DOC_ALIGN_MODE" = "gpt" ] && [ "$MANAGE_ALIGN_SERVER" = "1" ]; then
  read -r ALIGN_HOST ALIGN_PORT < <(pipeline_api_host_port "$DOC_ALIGN_API_BASE")
  ./scripts/stop_vllm.sh "$ALIGN_PORT" || true
  if [ "$ALIGN_HOST" = "localhost" ] || [ "$ALIGN_HOST" = "127.0.0.1" ]; then
    if command -v setsid >/dev/null 2>&1; then
      (setsid ./scripts/serve_vllm.sh "$DOC_ALIGN_MODEL_KEY" "$ALIGN_PORT") &
    else
      (./scripts/serve_vllm.sh "$DOC_ALIGN_MODEL_KEY" "$ALIGN_PORT") &
    fi
    ALIGN_PID=$!
    ./scripts/wait_server.sh "$DOC_ALIGN_API_BASE" 600
  fi
fi

for dataset in "${DATASET_LIST[@]}"; do
  BASE_DIR="$(pipeline_dataset_prepared_dir "$dataset")"
  mapfile -t LP_LIST < <(pipeline_list_lps "$dataset" "$LPS")
  if [ "${#LP_LIST[@]}" -eq 0 ]; then
    pipeline_log "No LPs found for dataset $dataset"
    continue
  fi

  DOC_DATASET="${dataset}${DOC_SUFFIX}"
  DOC_DIR="$(pipeline_dataset_prepared_dir "$DOC_DATASET")"

  for MODEL_KEY in "${MODEL_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      SENT_GEN="outputs/${RUN_NAME}/gen/${dataset}/${LP}/${MODEL_KEY}.jsonl"
      DOC_GEN="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}.jsonl"
      DOC_GEN_RAW="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__raw.jsonl"
      DOC_FROM_SENT="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__from_sent.jsonl"
      SENT_FROM_DOC="outputs/${RUN_NAME}/gen/${dataset}/${LP}/${MODEL_KEY}__from_doc.jsonl"

      if [ -f "$SENT_GEN" ]; then
        pipeline_docops to-doc \
          --input "$SENT_GEN" \
          --output "$DOC_FROM_SENT" \
          --sep "$DOC_GEN_SEP" \
          --fields "source,reference,hypothesis"
      else
        pipeline_log "Missing sentence gen: $SENT_GEN (skip from_sent)"
      fi

      if [ ! -f "$DOC_GEN" ]; then
        pipeline_log "Missing doc gen: $DOC_GEN (skip expand)"
        continue
      fi

      DOC_FOR_EXP="$DOC_GEN"
      if [ "$DOC_MARKER_ENABLE" = "1" ] && [ -f "$DOC_GEN_RAW" ]; then
        DOC_FOR_EXP="$DOC_GEN_RAW"
      fi
      SPLITTER="auto"
      if [ "$DOC_MARKER_ENABLE" = "1" ]; then
        SPLITTER="marker"
      fi

      pipeline_docops expand \
        --base "${BASE_DIR}/${LP}.jsonl" \
        --doc "$DOC_FOR_EXP" \
        --output "$SENT_FROM_DOC" \
        --sep "$DOC_SPLIT_SEP" \
        --splitter "$SPLITTER" \
        --marker-regex "$DOC_MARKER_REGEX" \
        --add-doc-hyp \
        --align-mode "$DOC_ALIGN_MODE" \
        $( [ "$DOC_ALIGN_META" = "1" ] && echo "--align-meta" ) \
        $( [ -n "$DOC_ALIGN_MODEL" ] && echo "--align-model $DOC_ALIGN_MODEL" ) \
        $( [ "$DOC_ALIGN_MODE" = "gpt" ] && echo "--align-api-base $DOC_ALIGN_API_BASE" ) \
        $( [ "$DOC_ALIGN_MODE" = "gpt" ] && echo "--align-model-name $DOC_ALIGN_MODEL_NAME" ) \
        $( [ "$DOC_ALIGN_MODE" = "gpt" ] && echo "--align-max-tokens $DOC_ALIGN_MAX_TOKENS" ) \
        $( [ "$DOC_ALIGN_MODE" = "gpt" ] && echo "--align-response-format $DOC_ALIGN_RESPONSE_FORMAT" )

      if [ "$DOC_MARKER_ENABLE" = "1" ]; then
        if [ "$DOC_MARKER_KEEP_RAW" = "1" ] && [ ! -f "$DOC_GEN_RAW" ]; then
          mv "$DOC_GEN" "$DOC_GEN_RAW"
        fi
        RAW_IN="$DOC_GEN"
        if [ -f "$DOC_GEN_RAW" ]; then
          RAW_IN="$DOC_GEN_RAW"
        fi
        pipeline_docops clean \
          --input "$RAW_IN" \
          --output "$DOC_GEN" \
          --marker-regex "$DOC_MARKER_REGEX" \
          --fields "source,reference,hypothesis"
      fi
    done
  done
done

if [ -n "$ALIGN_PID" ]; then
  kill -TERM -- "-$ALIGN_PID" 2>/dev/null || true
  kill -TERM "$ALIGN_PID" 2>/dev/null || true
  sleep 2
  if kill -0 "$ALIGN_PID" 2>/dev/null; then
    kill -KILL -- "-$ALIGN_PID" 2>/dev/null || true
    kill -KILL "$ALIGN_PID" 2>/dev/null || true
  fi
fi

if [ "$DOC_ALIGN_MODE" = "gpt" ] && [ "$MANAGE_ALIGN_SERVER" = "1" ]; then
  ./scripts/stop_vllm.sh "$ALIGN_PORT" || true
fi
