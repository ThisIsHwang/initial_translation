#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:?MODEL_KEY required (example: gpt_oss_120b)}"
PORT="${2:-8000}"

uv run evalmt-serve-vllm --model "$MODEL_KEY" --port "$PORT"
