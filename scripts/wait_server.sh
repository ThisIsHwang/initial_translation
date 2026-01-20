#!/usr/bin/env bash
set -euo pipefail

API_BASE="${1:-http://localhost:8000/v1}"
TIMEOUT="${2:-600}"

uv run evalmt-wait-server --api-base "$API_BASE" --timeout "$TIMEOUT"
