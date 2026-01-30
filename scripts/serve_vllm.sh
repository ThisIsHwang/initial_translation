#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:?MODEL_KEY required (example: gpt_oss_120b)}"
PORT="${2:-8000}"

# Optional: restrict GPUs used by the vLLM server.
# Example:
#   ./scripts/serve_vllm.sh gpt_oss_120b 8000 0,1,2,3,4,5,6,7
GPU_LIST="${3:-}"
UV_PROJECT_SERVE="${UV_PROJECT_SERVE:-${UV_PROJECT:-}}"

UV_ARGS=()
if [ -n "$UV_PROJECT_SERVE" ]; then
  UV_ARGS=(--project "$UV_PROJECT_SERVE")
fi

if [ -n "$GPU_LIST" ]; then
  echo "CUDA_VISIBLE_DEVICES=$GPU_LIST"
  exec env CUDA_VISIBLE_DEVICES="$GPU_LIST" uv run "${UV_ARGS[@]}" evalmt-serve-vllm --model "$MODEL_KEY" --port "$PORT"
else
  exec uv run "${UV_ARGS[@]}" evalmt-serve-vllm --model "$MODEL_KEY" --port "$PORT"
fi
