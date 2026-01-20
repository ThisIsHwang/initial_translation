#!/usr/bin/env bash
set -euo pipefail

# Install python dependencies into .venv using uv.
# This installs the current project too.
uv sync "$@"
