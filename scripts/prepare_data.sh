#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-wmt24pp}"
LPS="${2:-all}"  # "all" or "en-ko_KR,en-ja_JP"
OUT_DIR="${3:-data/${DATASET}}"

uv run evalmt-prepare --dataset "$DATASET" --lps "$LPS" --out "$OUT_DIR"
