#!/usr/bin/env bash

pipeline_log() {
  echo "[$(date +%H:%M:%S)] $*"
}

pipeline_die() {
  echo "ERROR: $*" >&2
  exit 1
}

pipeline_require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || pipeline_die "Missing required command: $cmd"
}

pipeline_normalize_sep() {
  local sep="$1"
  if [ "$sep" = "\\n" ]; then
    printf '\n'
  else
    printf '%s' "$sep"
  fi
}

pipeline_normalize_lp() {
  local lp="$1"
  while [[ "$lp" == *.jsonl || "$lp" == *.jsnol ]]; do
    lp="${lp%.*}"
  done
  printf '%s' "$lp"
}

pipeline_dataset_prepared_dir() {
  local dataset="$1"
  local cfg="configs/datasets/${dataset}.yaml"
  if [ -f "$cfg" ]; then
    local line
    line=$(rg -m1 '^prepared_dir:' "$cfg" 2>/dev/null || true)
    if [ -n "$line" ]; then
      echo "$line" | sed -E 's/^prepared_dir:[[:space:]]*//; s/^"//; s/"$//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^'\''//; s/'\''$//'
      return
    fi
  fi
  echo "data/$dataset"
}

pipeline_list_datasets() {
  local datasets="$1"
  if [ "$datasets" = "all" ]; then
    if [ -d configs/datasets ]; then
      ls configs/datasets/*.yaml 2>/dev/null | sed 's#.*/##; s/\\.yaml$//' | rg -v '_doc$' || true
    fi
    if [ -d data ]; then
      ls -1 data 2>/dev/null | rg -v '_doc$' || true
    fi
  else
    echo "${datasets//,/ }" | tr ' ' '\n' | sed '/^$/d'
  fi | sort -u
}

pipeline_list_models() {
  local models="$1"
  if [ "$models" = "all" ]; then
    ls configs/models/*.yaml 2>/dev/null | sed 's#.*/##; s/\\.yaml$//' || true
  else
    echo "${models//,/ }" | tr ' ' '\n' | sed '/^$/d'
  fi | sort -u
}

pipeline_list_metrics() {
  local metrics="$1"
  if [ "$metrics" = "all" ]; then
    ls configs/metrics/*.yaml 2>/dev/null | sed 's#.*/##; s/\\.yaml$//' || true
  else
    echo "${metrics//,/ }" | tr ' ' '\n' | sed '/^$/d'
  fi | sort -u
}

pipeline_list_lps() {
  local dataset="$1"
  local lps="$2"
  if [ "$lps" = "all" ]; then
    local dir
    dir=$(pipeline_dataset_prepared_dir "$dataset")
    if [ -d "$dir" ]; then
      ls "$dir"/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null | while read -r f; do
        pipeline_normalize_lp "$f"
        echo
      done | sed '/^$/d' | sort -u
    fi
  else
    for lp in ${lps//,/ }; do
      pipeline_normalize_lp "$lp"
      echo
    done | sed '/^$/d'
  fi
}

pipeline_api_host_port() {
  local api_base="$1"
  local host port
  host=$(echo "$api_base" | sed -E 's#^https?://([^/:]+).*#\\1#')
  port=$(echo "$api_base" | sed -E 's#^https?://[^:/]+:([0-9]+).*#\\1#')
  if [ -z "$port" ] || [ "$port" = "$api_base" ]; then
    port=8000
  fi
  printf '%s %s\n' "$host" "$port"
}

pipeline_uv_run() {
  local project="$1"
  shift
  if [ -n "$project" ]; then
    uv run --project "$project" "$@"
  else
    uv run "$@"
  fi
}

pipeline_project_for() {
  local kind="$1"
  case "$kind" in
    serve)
      echo "${UV_PROJECT_SERVE:-${UV_PROJECT:-}}"
      ;;
    gen)
      echo "${UV_PROJECT_GEN:-${UV_PROJECT:-}}"
      ;;
    align)
      echo "${UV_PROJECT_ALIGN:-${UV_PROJECT:-}}"
      ;;
    docops)
      echo "${UV_PROJECT_DOCOPS:-${UV_PROJECT_ALIGN:-${UV_PROJECT_GEN:-${UV_PROJECT:-}}}}"
      ;;
    score)
      echo "${UV_PROJECT_SCORE:-${UV_PROJECT:-}}"
      ;;
    *)
      echo "${UV_PROJECT:-}"
      ;;
  esac
}

pipeline_docops() {
  pipeline_uv_run "$(pipeline_project_for docops)" evalmt-docops "$@"
}

pipeline_jsonl_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "0"
    return
  fi
  python3 - "$path" <<'PY'
import sys
path = sys.argv[1]
cnt = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cnt += 1
print(cnt)
PY
}

pipeline_jsonl_has_key() {
  local path="$1"
  local key="$2"
  if [ ! -f "$path" ]; then
    echo "0"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$path" "$key" <<'PY'
import json, sys
path = sys.argv[1]
key = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            print(0)
            sys.exit(0)
        print(1 if key in row else 0)
        sys.exit(0)
print(0)
PY
  else
    if rg -q "\"$key\"" "$path"; then
      echo "1"
    else
      echo "0"
    fi
  fi
}

pipeline_metric_requires_reference() {
  local metric="$1"
  local cfg="configs/metrics/${metric}.yaml"
  if [ ! -f "$cfg" ]; then
    return 1
  fi
  if rg -q '^mode:\\s*ref' "$cfg"; then
    return 0
  fi
  if rg -q 'requires_reference:\\s*true' "$cfg"; then
    return 0
  fi
  return 1
}

pipeline_force_stop_vllm() {
  local ports=("$@")
  if [ "${#ports[@]}" -eq 0 ]; then
    ports=(8000 8001)
  fi
  for port in "${ports[@]}"; do
    if [ -x ./scripts/stop_vllm.sh ]; then
      ./scripts/stop_vllm.sh "$port" || true
    fi
  done
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "vllm" 2>/dev/null || true
  fi
}
