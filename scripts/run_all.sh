#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-run1}"
DATASET="${2:-wmt24pp}"
LPS="${3:-en-ko_KR}"  # "all" or "en-ko_KR,en-ja_JP"
MODELS="${4:-gpt_oss_120b}"
METRICS="${5:-xcomet_mqm,xcomet_qe,metricx24_ref,metricx24_qe}"
API_BASE="${6:-http://localhost:8000/v1}"

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a METRIC_LIST <<< "$METRICS"

# 1) Prepare data
./scripts/prepare_data.sh "$DATASET" "$LPS"

# 2) Determine LP list from prepared files
PREP_DIR="data/wmt24pp"
if [ "$DATASET" != "wmt24pp" ]; then
  PREP_DIR="data/$DATASET"
fi

mapfile -t LP_LIST < <(ls "$PREP_DIR" 2>/dev/null | grep -E '^en-.*\.jsonl$' | sed 's/\.jsonl$//')

if [ "${#LP_LIST[@]}" -eq 0 ]; then
  echo "No prepared language pairs found in $PREP_DIR"
  exit 1
fi

# 3) Loop models
for MODEL_KEY in "${MODEL_LIST[@]}"; do
  echo "=== Serving model: $MODEL_KEY ==="

  # Start server in background
  (./scripts/serve_vllm.sh "$MODEL_KEY" 8000) &
  SERVER_PID=$!

  # Ensure we stop server on exit
  cleanup() {
    echo "Stopping server PID=$SERVER_PID"
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  ./scripts/wait_server.sh "$API_BASE" 600

  # ------------------------------------------------------------
  # Phase A: generation while vLLM is running
  # ------------------------------------------------------------
  for LP in "${LP_LIST[@]}"; do
    echo "=== Generate: model=$MODEL_KEY lp=$LP ==="
    ./scripts/generate.sh "$RUN_NAME" "$DATASET" "$LP" "$MODEL_KEY" "$API_BASE"
  done

  # IMPORTANT (H100x8 / big models): metrics (COMET / MetricX) often require GPU.
  # If your vLLM server consumes all GPUs (e.g., TP=8), scoring while the server
  # is still running can OOM or contend for memory.
  # So we stop vLLM before scoring by default.
  cleanup
  trap - EXIT

  # ------------------------------------------------------------
  # Phase B: scoring after vLLM is stopped
  # ------------------------------------------------------------
  for METRIC in "${METRIC_LIST[@]}"; do
    for LP in "${LP_LIST[@]}"; do
      echo "=== Score: metric=$METRIC model=$MODEL_KEY lp=$LP ==="
      ./scripts/score.sh "$RUN_NAME" "$METRIC" "$DATASET" "$LP" "$MODEL_KEY"
    done
  done

done

./scripts/aggregate.sh "$RUN_NAME"
echo "✅ Done. See outputs/$RUN_NAME/summary.csv"
