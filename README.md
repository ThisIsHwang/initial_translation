# evalmt

A **maintainable MT (machine translation) evaluation repository** designed for:

- **Generation**: any model served via **vLLM** as an **OpenAI-compatible server** (`/v1/chat/completions`)
- **Required dataset**: **WMT24++** (`google/wmt24pp` on Hugging Face)
- **Required models** (provided as configs):
  - `gpt-oss-120b`
  - `qwen3-235b-a22b-instruct-2507`
  - `translategemma-4b-it`
  - `gemma-3-27b-it`
- **Required metrics** (provided as configs):
  - **XCOMET variants**
    - Reference-based (MQM-correlated)
    - QE (reference-free)
  - **MetricX**
    - Reference-based
    - QE (reference-free)

This repo is intentionally structured so that **data**, **models**, and **metrics** can be added by dropping in a new YAML config + a small, isolated Python runner.

---

## Design goals

1) **Config-driven**: almost all changes should happen in YAML under `configs/`.
2) **Stable intermediate format**: everything becomes a canonical JSONL schema early (`data/<dataset>/<lp>.jsonl`).
3) **Composable pipeline**:
   - Prepare dataset → Generate translations → Score metrics → Aggregate summaries
4) **Maintainability**:
   - small, readable modules
   - minimal “magic”
   - explicit output paths
   - careful schema evolution

---

## Table of contents

- [1. Quick start](#1-quick-start)
- [2. What this repo produces](#2-what-this-repo-produces)
- [3. Repo layout](#3-repo-layout)
- [4. Installation with uv](#4-installation-with-uv)
- [5. Installing vLLM](#5-installing-vllm)
- [5.3 H100 x8 single-node notes](#53-h100-x8-single-node-notes)
- [6. Installing MetricX code](#6-installing-metricx-code)
- [7. Dataset: WMT24++](#7-dataset-wmt24)
- [8. Run the pipeline](#8-run-the-pipeline)
- [9. Config system](#9-config-system)
- [10. Extending the repo](#10-extending-the-repo)
- [11. Output formats](#11-output-formats)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Maintenance guidelines](#13-maintenance-guidelines)

---

## 1. Quick start

### Requirements

- Linux is strongly recommended.
- Python **>= 3.10**
- Internet access is required to:
  - download WMT24++ from Hugging Face
  - download model weights
  - download COMET checkpoints
  - clone MetricX
- For large models (120B/235B): NVIDIA GPUs + CUDA, and **multiple GPUs** in most setups.

> Some model repos (e.g., Gemma family) may require accepting a license or logging into Hugging Face.

### One-time: install Python deps with **uv**

```bash
# from repo root
uv sync
```

This creates `.venv/` and installs:

- all Python dependencies from `pyproject.toml`
- the local package `evalmt` (so CLI commands become available)

Optional dev tools:

```bash
uv sync --extra dev
```

### Sanity check (optional)

If you want a quick environment check (uv + Python + optional vLLM/MetricX):

```bash
./scripts/doctor.sh
```

### Fetch MetricX code (required only if you will run MetricX metrics)

```bash
./scripts/fetch_metricx.sh
```

### Prepare WMT24++ data

Example: English → Korean

```bash
./scripts/prepare_data.sh wmt24pp en-ko_KR
```

All English→X pairs (auto-discovered from repo file list):

```bash
./scripts/prepare_data.sh wmt24pp all
```

### Serve a model with vLLM

In terminal A:

```bash
./scripts/serve_vllm.sh gpt_oss_120b 8000
```

In terminal B:

```bash
./scripts/wait_server.sh http://localhost:8000/v1
```

### Generate

```bash
./scripts/generate.sh run1 wmt24pp en-ko_KR gpt_oss_120b http://localhost:8000/v1
```

### Score

```bash
./scripts/score.sh run1 xcomet_mqm    wmt24pp en-ko_KR gpt_oss_120b
./scripts/score.sh run1 xcomet_qe     wmt24pp en-ko_KR gpt_oss_120b
./scripts/score.sh run1 metricx24_ref wmt24pp en-ko_KR gpt_oss_120b
./scripts/score.sh run1 metricx24_qe  wmt24pp en-ko_KR gpt_oss_120b
```

### Aggregate

```bash
./scripts/aggregate.sh run1
# outputs/run1/summary.csv
```

---

## 2. What this repo produces

The pipeline produces:

1) **Prepared dataset JSONL** per language pair

- Path: `data/<dataset>/<lp>.jsonl`
- Canonical fields: `id, lp, domain, source, reference, ...`

2) **Generation output JSONL** per model + language pair

- Path: `outputs/<run>/gen/<dataset>/<lp>/<model_key>.jsonl`
- Adds:
  - `hypothesis` (model translation)
  - `model` (the model key you passed)
  - `served_model` (the name vLLM exposes)
  - `gen_params` (temperature/top_p/max_tokens/stop)

3) **Metric output JSONL** per metric + model + language pair

- Path: `outputs/<run>/metrics/<metric_key>/<dataset>/<lp>/<model_key>.jsonl`
- Adds:
  - `metric` (metric key)
  - `score` (float)
  - optionally `error_spans` (for metrics that export them)

4) **Aggregate summary CSV**

- Path: `outputs/<run>/summary.csv`
- One row per `(dataset, lp, model, metric)` with `mean/min/max/n`.

---

## 3. Repo layout

```text
.
├── configs/
│   ├── datasets/
│   ├── models/
│   └── metrics/
├── evalmt/
│   ├── cli/
│   ├── datasets/
│   ├── generation/
│   ├── metrics/
│   └── utils/
├── scripts/
├── third_party/
│   └── metricx/            # created by scripts/fetch_metricx.sh
├── data/                   # prepared dataset outputs (gitignored)
└── outputs/                # generations + scores (gitignored)
```

**Key idea**: everything configurable lives under `configs/`.

- Add a model? Add `configs/models/<name>.yaml`.
- Add a metric? Add `configs/metrics/<name>.yaml`.
- Add a dataset loader? Add a dataset class + `configs/datasets/<name>.yaml`.

---

## 4. Installation with uv

### 4.1 Typical workflow

```bash
uv sync
```

Then run any command:

```bash
uv run python -c "import evalmt; print(evalmt.__version__)"
```

### 4.2 uv lock

For reproducibility in CI, teams often commit a lock file.

On a machine with internet:

```bash
uv lock
```

Then subsequent installs can be:

```bash
uv sync --frozen
```

---

## 5. Installing vLLM

vLLM installation is **GPU / CUDA / ROCm** specific.

This repo intentionally does **not** pin `vllm` in `pyproject.toml` because:

- the right wheel depends on your CUDA version
- `gpt-oss-120b` may require a special vLLM build

### 5.1 Standard vLLM (for many Hugging Face models)

If you already know what to install in your environment:

```bash
uv pip install vllm
```

### 5.2 gpt-oss-120b (special build)

The repo includes a script:

```bash
./scripts/install_vllm_gptoss.sh
```

If it fails, adjust versions / indexes according to your environment.

### 5.3 H100 x8 single-node notes

This repo is usable on many environments, but your stated target environment is:

- **1 node**
- **8× NVIDIA H100**

That changes a few practical defaults and "gotchas":

#### 5.3.1 Tensor parallel sizing

For your node, a reasonable starting point is:

- `gpt_oss_120b`: `tensor_parallel_size: 8`
- `qwen3_235b_a22b_instruct_2507`: `tensor_parallel_size: 8`
- `gemma3_27b_it`: `tensor_parallel_size: 1`
- `translategemma_4b_it`: `tensor_parallel_size: 1`

These are set in `configs/models/*.yaml` as the initial defaults.

If you want to change GPU usage:

- reduce/raise `tensor_parallel_size`
- optionally restrict visible GPUs when starting vLLM:

```bash
./scripts/serve_vllm.sh gpt_oss_120b 8000 0,1,2,3,4,5,6,7
```

#### 5.3.2 GPU contention between generation and scoring

**Important:** COMET/XCOMET and MetricX typically use the GPU.

If a vLLM server is using **all 8 GPUs** (e.g., TP=8 for a large model), running metric
scoring at the same time will often cause OOM or severe slowdown.

To keep things reliable, `scripts/run_all.sh` is structured into two phases per model:

1) start vLLM → **generate** all LPs
2) stop vLLM → **score** all metrics

That matches how you typically operate a single H100x8 node.

If you want to do more advanced GPU packing (e.g., small model on GPU0 while scoring on GPU7),
you can:

- start the server with an explicit GPU list (3rd arg in `serve_vllm.sh`)
- run scoring with a dedicated GPU using:

```bash
SCORE_GPU_LIST=7 ./scripts/score.sh ...
```

#### 5.3.3 Throughput tuning knobs

The most useful knobs for a high-throughput node:

- Increase HTTP client concurrency when generating:

```bash
CONCURRENCY=64 ./scripts/generate.sh run1 wmt24pp en-ko_KR gpt_oss_120b http://localhost:8000/v1
```

- Use `vllm.extra_args` in the model YAML to set vLLM flags (leave as empty list by default).

Always tune with a small LP first (e.g., `en-ko_KR`) before running `all`.

---

## 6. Installing MetricX code

MetricX is invoked by running its module entrypoints:

- `python -m metricx24.predict ...`

This repo keeps MetricX under `third_party/metricx` so that:

- the wrapper stays stable
- you can update MetricX by pulling upstream

Install:

```bash
./scripts/fetch_metricx.sh
```

Notes:

- MetricX requirements are installed with `uv pip install -r ...`.
- If your org vendors third-party code differently, update `evalmt/metrics/metricx_metric.py`.

---

## 7. Dataset: WMT24++

This repo fetches WMT24++ from Hugging Face via `huggingface_hub`.

### 7.1 Preparing language pairs

Single pair:

```bash
./scripts/prepare_data.sh wmt24pp en-ko_KR
```

All pairs (auto-discovered by listing repo files):

```bash
./scripts/prepare_data.sh wmt24pp all
```

Prepared files are written to:

- `data/wmt24pp/en-ko_KR.jsonl`

### 7.2 Filtering and reference choices

Dataset prep behavior is controlled by `configs/datasets/wmt24pp.yaml`:

- `filter_bad_source`: drop segments where the source is flagged as bad
- `use_post_edit_as_reference`: use the post-edited target as the reference

If you need to reproduce a specific evaluation protocol, pin these explicitly.

### 7.3 Caching tips

Hugging Face downloads can be large.

Useful environment variables:

- `HF_HOME` (cache root)
- `HF_HUB_ENABLE_HF_TRANSFER=1` (faster downloads on some systems)

---

## 8. Run the pipeline

### 8.1 Serving a model

```bash
./scripts/serve_vllm.sh gpt_oss_120b 8000
```

This:

- reads `configs/models/gpt_oss_120b.yaml`
- constructs a `vllm serve ...` command
- starts the server

### 8.2 Generation

```bash
./scripts/generate.sh run1 wmt24pp en-ko_KR gpt_oss_120b http://localhost:8000/v1
```

Generation is **resumable**:

- if output JSONL exists, previously generated `id`s are skipped

Tuning knobs:

- `--concurrency` (CLI flag) controls number of parallel requests
- `generation_defaults` in model YAML controls decoding

### 8.3 Scoring

```bash
./scripts/score.sh run1 xcomet_mqm wmt24pp en-ko_KR gpt_oss_120b
```

Scoring:

- loads generation JSONL
- runs the metric
- writes scored JSONL

### 8.4 Aggregation

```bash
./scripts/aggregate.sh run1
```

Produces:

- `outputs/run1/summary.csv`

### 8.5 Run everything (multi-model, multi-lp)

```bash
./scripts/run_all.sh run1 wmt24pp all \
  gpt_oss_120b,qwen3_235b_a22b_instruct_2507 \
  xcomet_mqm,xcomet_qe
```

`run_all.sh` will:

- prepare data
- loop models
  - start vLLM
  - generate for each LP
  - stop vLLM (to avoid GPU contention)
  - score for each metric
- aggregate

---

## 9. Config system

All configs are YAML in `configs/`.

### 9.1 Dataset config

Example: `configs/datasets/wmt24pp.yaml`

Fields:

- `type`: dataset loader type (maps to a class in `evalmt/datasets/`)
- `hf_repo`: HF dataset repo id
- `prepared_dir`: output directory for prepared JSONL
- dataset-specific toggles (e.g., filtering)

### 9.2 Model config

Example: `configs/models/gpt_oss_120b.yaml`

Fields:

- `hf_model_id`: passed to `vllm serve`
- `served_model_name`: name passed in OpenAI requests
- `vllm`: vLLM server parameters
- `prompt.system` and `prompt.user`: prompt templates
- `generation_defaults`: decoding defaults

**How prompts work**

- For each segment, we fill:
  - `{target_language}`
  - `{target_region}`
  - `{source}`

The target language/region is derived from the LP code (e.g., `en-ko_KR`).

### 9.3 Metric config

Example: `configs/metrics/xcomet_mqm.yaml`

Fields:

- `type`: metric runner type (`comet` or `metricx`)
- `mode`: `ref` or `qe`
- `direction`: `higher_is_better` or `lower_is_better`
- runner-specific parameters

**Important: metric direction**

- XCOMET / COMET scores: usually **higher is better**
- MetricX: usually an **error score**, so **lower is better**

Your aggregation can still compare means, but always respect `direction`.

---

## 10. Extending the repo

### 10.1 Add a new model

1) Add a YAML file under `configs/models/`.

2) Serve and generate:

```bash
./scripts/serve_vllm.sh <new_model_key> 8000
./scripts/generate.sh runX wmt24pp en-ko_KR <new_model_key> http://localhost:8000/v1
```

Tips:

- Set `tensor_parallel_size` to match your GPU topology.
- Prefer keeping prompts consistent across models for fair comparisons.
- If a model requires special chat formatting:
  - add model-specific logic to `evalmt/generation/vllm_openai.py`
  - or add a new generator module and keep the old path stable

### 10.2 Add a new metric

#### A) COMET / XCOMET model

Just add a config:

```yaml
type: comet
mode: ref
model: Unbabel/XCOMET-XXL
batch_size: 8
gpus: 1
```

No Python changes required.

#### B) External CLI metric

1) Implement a runner in `evalmt/metrics/` that:

- reads generation JSONL
- writes scored JSONL
- is driven entirely by YAML config

2) Register it using the decorator in `evalmt/metrics/registry.py`.

See `evalmt/metrics/metricx_metric.py` for an example.

### 10.3 Add a new dataset

1) Implement a dataset loader in `evalmt/datasets/`.

2) Register it with `@register_dataset("<type>")`.

3) Add config YAML under `configs/datasets/`.

**Rule of thumb**: convert everything into the canonical JSONL schema early.

---

## 11. Output formats

### 11.1 Prepared dataset JSONL

Each line is a JSON object. Minimal schema:

```json
{"id":"en-ko_KR:123","lp":"en-ko_KR","domain":"news","source":"...","reference":"..."}
```

### 11.2 Generation JSONL

Adds:

- `hypothesis`: model translation
- `model`: model key
- `served_model`: served name on vLLM
- `gen_params`: decoding params

### 11.3 Metric JSONL

Adds:

- `metric`: metric key
- `score`: float

Some metrics may also add:

- `error_spans`

---

## 12. Troubleshooting

### vLLM server never becomes ready

Common fixes:

- Reduce `gpu_memory_utilization` in the model YAML.
- Reduce `max_model_len`.
- Reduce `tensor_parallel_size` if misconfigured.
- Ensure you installed the correct vLLM build (especially for `gpt-oss`).

### MetricX errors: module not found

- Ensure `third_party/metricx` exists.
- Run `./scripts/fetch_metricx.sh`.

### COMET model download issues

- COMET downloads weights on first run.
- Ensure internet access and enough disk.

### Output contains extra wrappers ("Translation:", code fences, etc.)

- Cleanup rules are in `evalmt/generation/vllm_openai.py`.
- Add additional patterns there if you see systematic wrapper text.

### Hugging Face auth / gated repos

If your model/dataset is gated:

```bash
huggingface-cli login
```

---

## 13. Maintenance guidelines

### 13.1 Keep configs the source of truth

- Prefer YAML changes over hardcoding.
- Keep new config keys backward-compatible.

### 13.2 Avoid breaking schemas

Downstream tools assume stable fields:

- prepared dataset rows: `id, source, reference`
- generation rows: plus `hypothesis`
- metric rows: plus `score`

If you must change a schema:

- add new fields first
- keep old fields for at least one release cycle
- document the change in README or CHANGELOG

### 13.3 Reproducibility

- Keep `temperature=0` for deterministic decoding.
- Store decoding params in output JSONL (`gen_params`).
- Pin tool versions using `uv lock`.

### 13.4 Updating third-party tools

- MetricX: `cd third_party/metricx && git pull`
- COMET: upgrading `unbabel-comet` may change model loading behavior; test before bumping.

---
