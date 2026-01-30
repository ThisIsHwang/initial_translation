#!/usr/bin/env bash
set -euo pipefail

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

if [ "$DOC_GEN_SEP" = "\\n" ]; then
  DOC_GEN_SEP=$'\n'
fi
if [ "$DOC_SPLIT_SEP" = "\\n" ]; then
  DOC_SPLIT_SEP=$'\n'
fi

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

resolve_datasets
resolve_models

if [ "$CLEAN_GPU" = "1" ]; then
  ./scripts/clean_gpu.sh
fi

uv_run_docops() {
  local project="${UV_PROJECT_DOCOPS:-${UV_PROJECT_ALIGN:-${UV_PROJECT:-}}}"
  if [ -n "$project" ]; then
    uv run --project "$project" "$@"
  else
    uv run "$@"
  fi
}

ALIGN_PID=""
if [ "$DOC_ALIGN_MODE" = "gpt" ] && [ "$MANAGE_ALIGN_SERVER" = "1" ]; then
  ALIGN_HOST=$(echo "$DOC_ALIGN_API_BASE" | sed -E 's#^https?://([^/:]+).*#\\1#')
  ALIGN_PORT=$(echo "$DOC_ALIGN_API_BASE" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
  if [ -z "$ALIGN_PORT" ] || [ "$ALIGN_PORT" = "$DOC_ALIGN_API_BASE" ]; then
    ALIGN_PORT=8001
  fi
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
  read -r -a LP_LIST <<< "$(get_lp_list "$dataset")"
  if [ "${#LP_LIST[@]}" -eq 0 ]; then
    echo "No LPs found for dataset $dataset"
    continue
  fi

  DOC_DATASET="${dataset}${DOC_SUFFIX}"

  for MODEL_KEY in "${MODEL_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      SENT_GEN="outputs/${RUN_NAME}/gen/${dataset}/${LP}/${MODEL_KEY}.jsonl"
      DOC_GEN="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}.jsonl"
      DOC_GEN_RAW="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__raw.jsonl"
      DOC_FROM_SENT="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__from_sent.jsonl"
      SENT_FROM_DOC="outputs/${RUN_NAME}/gen/${dataset}/${LP}/${MODEL_KEY}__from_doc.jsonl"

      if [ -f "$SENT_GEN" ]; then
        uv_run_docops evalmt-docops to-doc \
          --input "$SENT_GEN" \
          --output "$DOC_FROM_SENT" \
          --sep "$DOC_GEN_SEP" \
          --fields "source,reference,hypothesis"
      else
        echo "Missing sentence gen: $SENT_GEN (skip from_sent)"
      fi

      if [ ! -f "$DOC_GEN" ]; then
        echo "Missing doc gen: $DOC_GEN (skip expand)"
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

      uv_run_docops evalmt-docops expand \
        --base "data/${dataset}/${LP}.jsonl" \
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
        uv_run_docops evalmt-docops clean \
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
