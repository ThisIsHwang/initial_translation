#!/usr/bin/env bash
set -euo pipefail

# Best-effort GPU cleanup before scoring.
# Skips if nvidia-smi is unavailable or no compute processes are found.

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; skipping GPU cleanup."
  exit 0
fi

USER_NAME="$(id -un)"
EXCLUDE_PIDS="${GPU_CLEAN_EXCLUDE_PIDS:-}"

mapfile -t PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u)

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No GPU compute processes found."
  exit 0
fi

KILL_LIST=()
for pid in "${PIDS[@]}"; do
  if [ -n "$EXCLUDE_PIDS" ] && [[ ",$EXCLUDE_PIDS," == *",$pid,"* ]]; then
    continue
  fi
  owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
  if [ -n "$owner" ] && [ "$owner" = "$USER_NAME" ]; then
    KILL_LIST+=("$pid")
  fi
done

if [ "${#KILL_LIST[@]}" -eq 0 ]; then
  echo "No GPU compute processes owned by $USER_NAME to kill."
  exit 0
fi

echo "Killing GPU compute PIDs: ${KILL_LIST[*]}"
kill -TERM "${KILL_LIST[@]}" 2>/dev/null || true
sleep 2
for pid in "${KILL_LIST[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
done
