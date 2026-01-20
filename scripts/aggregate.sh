#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?RUN_NAME required}"

uv run evalmt-aggregate --run "$RUN_NAME"
