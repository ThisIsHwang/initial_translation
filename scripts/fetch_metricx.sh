#!/usr/bin/env bash
set -euo pipefail

# Fetch MetricX source into third_party/metricx and install its requirements using uv.

METRICX_DIR="third_party/metricx"

mkdir -p third_party

if [ ! -d "$METRICX_DIR/.git" ] && [ ! -d "$METRICX_DIR" ]; then
  echo "Cloning MetricX into $METRICX_DIR ..."
  git clone https://github.com/google-research/metricx.git "$METRICX_DIR"
else
  echo "MetricX already exists at $METRICX_DIR"
fi

if [ -f "$METRICX_DIR/requirements.txt" ]; then
  echo "Installing MetricX requirements via uv ..."
  uv pip install -r "$METRICX_DIR/requirements.txt"
else
  echo "WARNING: $METRICX_DIR/requirements.txt not found. Check MetricX repo layout."
fi

echo "✅ MetricX ready."
