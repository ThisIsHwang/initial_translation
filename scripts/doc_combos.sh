#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"
DATASET="${2:?DATASET required}"
LPS="${3:?LPS required (ex: en-ko_KR or en-ko_KR,en-ja_JP)}"
MODELS="${4:?MODELS required (comma-separated)}"
METRICS="${5:-xcomet_mqm,xcomet_qe,metricx24_ref,metricx24_qe,bleu}"
API_BASE="${6:-http://localhost:8000/v1}"

DOC_SUFFIX="${DOC_SUFFIX:-_doc}"
DOC_GEN_SEP="${DOC_GEN_SEP:-$'\n'}"
DOC_SPLIT_SEP="${DOC_SPLIT_SEP:-$DOC_GEN_SEP}"

# Allow common literal escape
if [ "$DOC_GEN_SEP" = "\\n" ]; then
  DOC_GEN_SEP=$'\n'
fi
if [ "$DOC_SPLIT_SEP" = "\\n" ]; then
  DOC_SPLIT_SEP=$'\n'
fi

DOC_DATASET="${DATASET}${DOC_SUFFIX}"
DOC_PREP_DIR="data/${DOC_DATASET}"

IFS=',' read -r -a LP_LIST <<< "$LPS"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a METRIC_LIST <<< "$METRICS"

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
  uv run evalmt-docops to-doc \
    --input "$BASE_PATH" \
    --output "$DOC_PATH" \
    --sep "$DOC_GEN_SEP" \
    --fields "source,reference"
done

for MODEL_KEY in "${MODEL_LIST[@]}"; do
  echo "=== Serving model: $MODEL_KEY ==="
  HOST=$(echo "$API_BASE" | sed -E 's#^https?://([^/:]+).*#\\1#')
  PORT=$(echo "$API_BASE" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
  if [ -z "$PORT" ] || [ "$PORT" = "$API_BASE" ]; then
    PORT=8000
  fi

  SERVER_PID=""
  cleanup() {
    if [ -n "$SERVER_PID" ]; then
      echo "Stopping server PID=$SERVER_PID"
      kill "$SERVER_PID" 2>/dev/null || true
    fi
  }

  if [ "$HOST" = "localhost" ] || [ "$HOST" = "127.0.0.1" ]; then
    (./scripts/serve_vllm.sh "$MODEL_KEY" "$PORT") &
    SERVER_PID=$!
    trap cleanup EXIT
    ./scripts/wait_server.sh "$API_BASE" 600
  else
    echo "API_BASE is remote ($HOST); skipping local serve_vllm."
  fi

  # Sentence-level translation
  for LP in "${LP_LIST[@]}"; do
    ./scripts/generate.sh "$RUN_NAME" "$DATASET" "$LP" "$MODEL_KEY" "$API_BASE"
  done

  # Doc-level translation
  for LP in "${LP_LIST[@]}"; do
    ./scripts/generate.sh "$RUN_NAME" "$DOC_DATASET" "$LP" "$MODEL_KEY" "$API_BASE"
  done

  # Derive doc-gen from sentence-gen + expand doc-gen to sentence
  for LP in "${LP_LIST[@]}"; do
    SENT_GEN="outputs/${RUN_NAME}/gen/${DATASET}/${LP}/${MODEL_KEY}.jsonl"
    DOC_GEN="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}.jsonl"
    DOC_FROM_SENT="outputs/${RUN_NAME}/gen/${DOC_DATASET}/${LP}/${MODEL_KEY}__from_sent.jsonl"
    SENT_FROM_DOC="outputs/${RUN_NAME}/gen/${DATASET}/${LP}/${MODEL_KEY}__from_doc.jsonl"

    uv run evalmt-docops to-doc \
      --input "$SENT_GEN" \
      --output "$DOC_FROM_SENT" \
      --sep "$DOC_GEN_SEP" \
      --fields "source,reference,hypothesis"

    uv run evalmt-docops expand \
      --base "data/${DATASET}/${LP}.jsonl" \
      --doc "$DOC_GEN" \
      --output "$SENT_FROM_DOC" \
      --sep "$DOC_SPLIT_SEP" \
      --add-doc-hyp
  done

  # Stop vLLM before scoring (avoid GPU contention)
  cleanup
  trap - EXIT || true

  # Scoring: 4 combos
  for METRIC in "${METRIC_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      # sent -> sent
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "$MODEL_KEY"
      # sent -> doc
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DOC_DATASET" "$LP" "${MODEL_KEY}__from_sent"
      # doc -> doc
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DOC_DATASET" "$LP" "$MODEL_KEY"
      # doc -> sent
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "${MODEL_KEY}__from_doc"
    done
  done
done

./scripts/aggregate.sh "$RUN_NAME"
uv run evalmt-aggregate-combos --run "$RUN_NAME"
echo "✅ Done. See outputs/$RUN_NAME/summary.csv and summary_combos.csv"
