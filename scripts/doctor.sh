#!/usr/bin/env bash
set -euo pipefail

echo "== evalmt doctor =="

echo "[1/7] uv version"
uv --version || { echo "uv not found"; exit 1; }

echo

echo "[2/7] Python version (inside uv env)"
uv run python -V

echo

echo "[3/7] evalmt import"
uv run python -c "import evalmt; print('evalmt version:', getattr(evalmt, '__version__', 'unknown'))"

echo

echo "[4/7] vLLM import (optional)"
if uv run python -c "import vllm" >/dev/null 2>&1; then
  echo "vllm: OK"
else
  echo "vllm: NOT INSTALLED (this is fine until you serve a model)"
fi

echo

echo "[5/7] MetricX checkout (optional)"
if [ -d "third_party/metricx" ]; then
  echo "third_party/metricx: present"
else
  echo "third_party/metricx: missing (run ./scripts/fetch_metricx.sh if you need MetricX)"
fi

echo

echo "[6/7] Writable outputs"
mkdir -p outputs data
[ -w outputs ] && echo "outputs/: writable" || echo "outputs/: NOT writable"
[ -w data ] && echo "data/: writable" || echo "data/: NOT writable"

echo

echo "[7/7] GPU info (optional)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "nvidia-smi not found"
fi

echo
echo "✅ doctor finished"
