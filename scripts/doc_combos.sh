#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"
DATASET="${2:?DATASET required}"
LPS="${3:?LPS required (ex: en-ko_KR or en-ko_KR,en-ja_JP)}"
MODELS="${4:?MODELS required (comma-separated)}"
METRICS="${5:-xcomet_mqm,xcomet_qe,metricx24_ref,metricx24_qe,bleu}"
API_BASE="${6:-http://localhost:8000/v1}"
MANAGE_SERVER="${MANAGE_SERVER:-0}"

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
DOC_ALIGN_API_BASE="${DOC_ALIGN_API_BASE:-$API_BASE}"
DOC_ALIGN_MODEL_KEY="${DOC_ALIGN_MODEL_KEY:-gpt_oss_120b}"
DOC_ALIGN_MODEL_NAME="${DOC_ALIGN_MODEL_NAME:-gpt-oss-120b}"
DOC_ALIGN_MAX_TOKENS="${DOC_ALIGN_MAX_TOKENS:-64000}"
MANAGE_ALIGN_SERVER="${MANAGE_ALIGN_SERVER:-0}"
DOC_ALIGN_RESPONSE_FORMAT="${DOC_ALIGN_RESPONSE_FORMAT:-json_schema}"

# Allow common literal escape
if [ "$DOC_GEN_SEP" = "\\n" ]; then
  DOC_GEN_SEP=$'\n'
fi
if [ "$DOC_SPLIT_SEP" = "\\n" ]; then
  DOC_SPLIT_SEP=$'\n'
fi

# Count non-empty jsonl lines (best-effort)
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

# Check if generation is needed for a model across all LPs
needs_generation() {
  local model_key="$1"
  local need=0
  for lp in "${LP_LIST[@]}"; do
    local base_path="data/${DATASET}/${lp}.jsonl"
    local doc_path="data/${DOC_DATASET}/${lp}.jsonl"
    local sent_gen="outputs/${RUN_NAME}/gen/${DATASET}/${lp}/${model_key}.jsonl"
    local doc_gen="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${lp}/${model_key}.jsonl"

    local base_n
    local doc_n
    base_n=$(jsonl_count "$base_path")
    doc_n=$(jsonl_count "$doc_path")

    local sent_n
    local doc_n_gen
    sent_n=$(jsonl_count "$sent_gen")
    doc_n_gen=$(jsonl_count "$doc_gen")

    if [ "$sent_n" -lt "$base_n" ] || [ "$doc_n_gen" -lt "$doc_n" ]; then
      need=1
      break
    fi
  done
  echo "$need"
}

DOC_DATASET="${DATASET}${DOC_SUFFIX}"
DOC_PREP_DIR="data/${DOC_DATASET}"

IFS=',' read -r -a LP_LIST <<< "$LPS"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a METRIC_LIST <<< "$METRICS"

# Split metrics: sentence-eval uses non-context; doc-eval uses context (if provided).
# Metrics without context (e.g., BLEU/MetricX) are evaluated for both.
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

# Build doc-level datasets
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
  echo "=== Serving model: $MODEL_KEY ==="
  HOST=$(echo "$API_BASE" | sed -E 's#^https?://([^/:]+).*#\\1#')
  PORT=$(echo "$API_BASE" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
  if [ -z "$PORT" ] || [ "$PORT" = "$API_BASE" ]; then
    PORT=8000
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
    # Sentence-level translation
    for LP in "${LP_LIST[@]}"; do
      ./scripts/generate.sh "$RUN_NAME" "$DATASET" "$LP" "$MODEL_KEY" "$API_BASE"
    done

    # Doc-level translation
    for LP in "${LP_LIST[@]}"; do
      ./scripts/generate.sh "$RUN_NAME" "$DOC_DATASET" "$LP" "$MODEL_KEY" "$API_BASE"
    done
  else
    echo "All generation outputs already exist for $MODEL_KEY; skipping generation."
  fi

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
      (setsid ./scripts/serve_vllm.sh "$DOC_ALIGN_MODEL_KEY" "$ALIGN_PORT") &
    else
      (./scripts/serve_vllm.sh "$DOC_ALIGN_MODEL_KEY" "$ALIGN_PORT") &
      fi
      ALIGN_PID=$!
      ./scripts/wait_server.sh "$DOC_ALIGN_API_BASE" 600
    fi
  fi

  # Derive doc-gen from sentence-gen + expand doc-gen to sentence (doc->sent only)
  for LP in "${LP_LIST[@]}"; do
    SENT_GEN="outputs/${RUN_NAME}/gen/${DATASET}/${LP}/${MODEL_KEY}.jsonl"
    DOC_GEN="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}.jsonl"
    DOC_GEN_RAW="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__raw.jsonl"
    DOC_FROM_SENT="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__from_sent.jsonl"
    SENT_FROM_DOC="outputs/${RUN_NAME}/gen/${DATASET}/${LP}/${MODEL_KEY}__from_doc.jsonl"

    uv run evalmt-docops to-doc \
      --input "$SENT_GEN" \
      --output "$DOC_FROM_SENT" \
      --sep "$DOC_GEN_SEP" \
      --fields "source,reference,hypothesis"

    DOC_FOR_EXP="$DOC_GEN"
    if [ "$DOC_MARKER_ENABLE" = "1" ] && [ -f "$DOC_GEN_RAW" ]; then
      DOC_FOR_EXP="$DOC_GEN_RAW"
    fi
    SPLITTER="auto"
    if [ "$DOC_MARKER_ENABLE" = "1" ]; then
      SPLITTER="marker"
    fi

    uv run evalmt-docops expand \
      --base "data/${DATASET}/${LP}.jsonl" \
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
      if [ -f "$DOC_GEN" ]; then
        if [ "$DOC_MARKER_KEEP_RAW" = "1" ] && [ ! -f "$DOC_GEN_RAW" ]; then
          mv "$DOC_GEN" "$DOC_GEN_RAW"
        fi
        RAW_IN="$DOC_GEN"
        if [ -f "$DOC_GEN_RAW" ]; then
          RAW_IN="$DOC_GEN_RAW"
        fi
        uv run evalmt-docops clean \
          --input "$RAW_IN" \
          --output "$DOC_GEN" \
          --marker-regex "$DOC_MARKER_REGEX" \
          --fields "source,reference,hypothesis"
      fi
    fi
  done

  # Scoring: 4 combos
  for LP in "${LP_LIST[@]}"; do
    SENT_FROM_DOC="outputs/${RUN_NAME}/gen/${DATASET}/${LP}/${MODEL_KEY}__from_doc.jsonl"

    # sentence evals (non-context)
    for METRIC in "${METRICS_SENT[@]}"; do
      # sent -> sent
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "$MODEL_KEY"
      # doc -> sent
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "${MODEL_KEY}__from_doc"
    done

    # document evals (context)
    for METRIC in "${METRICS_DOC[@]}"; do
      # sent -> doc (context on sentence data)
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "$MODEL_KEY"
      # doc -> doc
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "${MODEL_KEY}__from_doc"
    done

    # document evals (non-context) for doc input -> doc eval
    for METRIC in "${METRICS_SENT[@]}"; do
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DOC_DATASET" "$LP" "$MODEL_KEY"
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
