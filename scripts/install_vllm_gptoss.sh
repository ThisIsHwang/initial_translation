#!/usr/bin/env bash
set -euo pipefail

# Install the gpt-oss-compatible vLLM build using uv.
# NOTE: Exact versions / indexes may change. If this fails, consult the gpt-oss model card.

uv pip install --pre "vllm==0.10.1+gptoss" \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

echo "✅ vLLM (gpt-oss flavor) installed."
