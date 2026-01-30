#!/usr/bin/env bash
set -euo pipefail

PORT="${1:?PORT required}"

PIDS=""
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
elif command -v fuser >/dev/null 2>&1; then
  PIDS=$(fuser -n tcp "$PORT" 2>/dev/null || true)
elif command -v pgrep >/dev/null 2>&1; then
  PIDS=$(pgrep -f "evalmt-serve-vllm.*--port[ =]$PORT" 2>/dev/null || true)
fi

if [ -z "$PIDS" ]; then
  exit 0
fi

for pid in $PIDS; do
  echo "Stopping vLLM PID=$pid on port $PORT"
  kill -TERM -- "-$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
done

sleep 2

for pid in $PIDS; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  fi
done
