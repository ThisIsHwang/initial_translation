#!/usr/bin/env bash
set -euo pipefail

echo "== evalmt doctor =="

echo "[1/6] uv version"
uv --version || { echo "uv not found"; exit 1; }

echo

echo "[2/6] Python version (inside uv env)"
uv run python -V

echo

echo "[3/6] evalmt import"
uv run python -c "import evalmt; print('evalmt version:', getattr(evalmt, '__version__', 'unknown'))"

echo

echo "[4/6] vLLM import (optional)"
if uv run python -c "import vllm" >/dev/null 2>&1; then
  echo "vllm: OK"
else
  echo "vllm: NOT INSTALLED (this is fine until you serve a model)"
fi

echo

echo "[5/6] MetricX checkout (optional)"
if [ -d "third_party/metricx" ]; then
  echo "third_party/metricx: present"
else
  echo "third_party/metricx: missing (run ./scripts/fetch_metricx.sh if you need MetricX)"
fi

echo

echo "[6/6] Writable outputs"
mkdir -p outputs data
[ -w outputs ] && echo "outputs/: writable" || echo "outputs/: NOT writable"
[ -w data ] && echo "data/: writable" || echo "data/: NOT writable"

echo
echo "✅ doctor finished"
