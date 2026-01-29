#!/usr/bin/env bash
set -euo pipefail

# Run doc-level translation for reference50, then split to sentences and score with context.

RUN_NAME="${1:?RUN_NAME required}"
API_BASE="${2:-http://localhost:8000/v1}"

DATASET="reference50"
DOC_SUFFIX="_doc"
DOC_DATASET="${DATASET}${DOC_SUFFIX}"

LPS_DEFAULT="en-ja_JP,en-ko_KR,en-vi_VN,en-zh_CN,ja_JP-en,ko_KR-en,vi_VN-en,zh_CN-en"
MODELS_DEFAULT="gpt_oss_120b,translategemma_27b_it,gemma3_27b_it"
METRICS_DEFAULT="xcomet_mqm_ctx"

LPS="${LPS:-$LPS_DEFAULT}"
MODELS="${MODELS:-$MODELS_DEFAULT}"
METRICS="${METRICS:-$METRICS_DEFAULT}"

MANAGE_SERVER="${MANAGE_SERVER:-0}"

DOC_GEN_SEP="${DOC_GEN_SEP:-$'\n'}"
DOC_SPLIT_SEP="${DOC_SPLIT_SEP:-$DOC_GEN_SEP}"

DOC_MARKER_ENABLE="${DOC_MARKER_ENABLE:-1}"
DOC_MARKER_TEMPLATE="${DOC_MARKER_TEMPLATE:-⟦{i}⟧}"
DOC_MARKER_JOIN="${DOC_MARKER_JOIN:- }"
DOC_MARKER_FIELDS="${DOC_MARKER_FIELDS:-source}"
DOC_MARKER_REGEX="${DOC_MARKER_REGEX:-⟦\\d+⟧}"
DOC_ALIGN_MODE="${DOC_ALIGN_MODE:-rule}"
DOC_ALIGN_META="${DOC_ALIGN_META:-0}"
DOC_ALIGN_MODEL="${DOC_ALIGN_MODEL:-}"
DOC_ALIGN_API_BASE="${DOC_ALIGN_API_BASE:-$API_BASE}"
DOC_ALIGN_MODEL_NAME="${DOC_ALIGN_MODEL_NAME:-gpt_oss_120b}"
MANAGE_ALIGN_SERVER="${MANAGE_ALIGN_SERVER:-0}"
DOC_ALIGN_RESPONSE_FORMAT="${DOC_ALIGN_RESPONSE_FORMAT:-json_schema}"

if [ "$DOC_GEN_SEP" = "\\n" ]; then
  DOC_GEN_SEP=$'\n'
fi
if [ "$DOC_SPLIT_SEP" = "\\n" ]; then
  DOC_SPLIT_SEP=$'\n'
fi

IFS=',' read -r -a LP_LIST <<< "$LPS"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a METRIC_LIST <<< "$METRICS"

DOC_PREP_DIR="data/${DOC_DATASET}"
mkdir -p "$DOC_PREP_DIR"

# Create doc dataset config if missing (generate uses prepared_dir)
DOC_CFG="configs/datasets/${DOC_DATASET}.yaml"
if [ ! -f "$DOC_CFG" ]; then
  cat > "$DOC_CFG" <<EOF
type: ${DATASET}
prepared_dir: ${DOC_PREP_DIR}
EOF
  echo "Wrote $DOC_CFG"
fi

# Build doc-level datasets with marker in source (for alignment)
for LP in "${LP_LIST[@]}"; do
  BASE_PATH="data/${DATASET}/${LP}.jsonl"
  if [ ! -f "$BASE_PATH" ]; then
    echo "Missing base dataset: $BASE_PATH"
    exit 1
  fi
  DOC_PATH="${DOC_PREP_DIR}/${LP}.jsonl"
  if [ "$DOC_MARKER_ENABLE" = "1" ]; then
    uv run evalmt-docops to-doc \
      --input "$BASE_PATH" \
      --output "$DOC_PATH" \
      --sep "$DOC_GEN_SEP" \
      --fields "source,reference" \
      --marker-template "$DOC_MARKER_TEMPLATE" \
      --marker-join "$DOC_MARKER_JOIN" \
      --marker-fields "$DOC_MARKER_FIELDS"
  else
    uv run evalmt-docops to-doc \
      --input "$BASE_PATH" \
      --output "$DOC_PATH" \
      --sep "$DOC_GEN_SEP" \
      --fields "source,reference"
  fi
done

for MODEL_KEY in "${MODEL_LIST[@]}"; do
  echo "=== Model: $MODEL_KEY ==="
  HOST=$(echo "$API_BASE" | sed -E 's#^https?://([^/:]+).*#\\1#')
  PORT=$(echo "$API_BASE" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
  if [ -z "$PORT" ] || [ "$PORT" = "$API_BASE" ]; then
    PORT=8000
  fi

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

  if [ "$MANAGE_SERVER" = "1" ] && { [ "$HOST" = "localhost" ] || [ "$HOST" = "127.0.0.1" ]; }; then
    if command -v setsid >/dev/null 2>&1; then
      (setsid ./scripts/serve_vllm.sh "$MODEL_KEY" "$PORT") &
    else
      (./scripts/serve_vllm.sh "$MODEL_KEY" "$PORT") &
    fi
    SERVER_PID=$!
    trap cleanup EXIT
    ./scripts/wait_server.sh "$API_BASE" 600
  fi

  # Doc-level translation
  for LP in "${LP_LIST[@]}"; do
    ./scripts/generate.sh "$RUN_NAME" "$DOC_DATASET" "$LP" "$MODEL_KEY" "$API_BASE"
  done

  # Stop vLLM before alignment/splitting (avoid GPU contention)
  cleanup
  trap - EXIT || true

  if [ "${CLEAN_GPU:-1}" = "1" ]; then
    ./scripts/clean_gpu.sh
  fi

  # Optional: start a dedicated GPT alignment server (default model: gpt_oss_120b)
  ALIGN_PID=""
  if [ "$DOC_ALIGN_MODE" = "gpt" ] && [ "$MANAGE_ALIGN_SERVER" = "1" ]; then
    ALIGN_HOST=$(echo "$DOC_ALIGN_API_BASE" | sed -E 's#^https?://([^/:]+).*#\\1#')
    ALIGN_PORT=$(echo "$DOC_ALIGN_API_BASE" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
    if [ -z "$ALIGN_PORT" ] || [ "$ALIGN_PORT" = "$DOC_ALIGN_API_BASE" ]; then
      ALIGN_PORT=8001
    fi
    if [ "$ALIGN_HOST" = "localhost" ] || [ "$ALIGN_HOST" = "127.0.0.1" ]; then
      if command -v setsid >/dev/null 2>&1; then
        (setsid ./scripts/serve_vllm.sh "$DOC_ALIGN_MODEL_NAME" "$ALIGN_PORT") &
      else
        (./scripts/serve_vllm.sh "$DOC_ALIGN_MODEL_NAME" "$ALIGN_PORT") &
      fi
      ALIGN_PID=$!
      ./scripts/wait_server.sh "$DOC_ALIGN_API_BASE" 600
    fi
  fi

  # Split doc translation back to sentences for scoring
  for LP in "${LP_LIST[@]}"; do
    DOC_GEN="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}.jsonl"
    SENT_FROM_DOC="outputs/${RUN_NAME}/gen/${DATASET}/${LP}/${MODEL_KEY}__from_doc.jsonl"

    SPLITTER="auto"
    if [ "$DOC_MARKER_ENABLE" = "1" ]; then
      SPLITTER="auto"
    fi

    uv run evalmt-docops expand \
      --base "data/${DATASET}/${LP}.jsonl" \
      --doc "$DOC_GEN" \
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
      $( [ "$DOC_ALIGN_MODE" = "gpt" ] && echo "--align-response-format $DOC_ALIGN_RESPONSE_FORMAT" )
  done

  # Context scoring on sentence-level split outputs
  for METRIC in "${METRIC_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "${MODEL_KEY}__from_doc"
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
done

./scripts/aggregate.sh "$RUN_NAME"
uv run evalmt-aggregate-combos --run "$RUN_NAME"
echo "✅ Done. See outputs/$RUN_NAME/summary.csv and summary_combos.csv"
