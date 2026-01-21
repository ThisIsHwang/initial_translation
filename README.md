# evalmt

vLLM 기반 **기계번역(MT) 생성 + 평가 파이프라인**입니다.  
WMT24++ 데이터셋을 준비하고, vLLM(OpenAI 호환 API)로 번역을 생성한 뒤, XCOMET/MetricX로 점수화하고 요약을 집계합니다.

핵심 포인트:
- **YAML 기반 구성**: 데이터/모델/메트릭은 `configs/`에 정의
- **표준 JSONL 스키마**: 모든 결과는 안정적인 JSONL 형식으로 유지
- **재현성/유지보수**: 작은 모듈 + 명시적 출력 경로
- **OpenAI 호환 서버**: `/v1/chat/completions` 엔드포인트 기반

---

## 목차

- [1. 빠른 시작](#1-빠른-시작)
- [2. 결과물 개요](#2-결과물-개요)
- [3. 레포 구조](#3-레포-구조)
- [4. 설치 (uv)](#4-설치-uv)
- [5. vLLM 설치](#5-vllm-설치)
- [6. MetricX 설치](#6-metricx-설치)
- [7. 데이터 준비 (WMT24++)](#7-데이터-준비-wmt24)
- [8. 파이프라인 실행](#8-파이프라인-실행)
- [9. 스크립트/CLI 요약](#9-스크립트cli-요약)
- [10. 구성 시스템](#10-구성-시스템)
- [11. 환경 변수/튜닝](#11-환경-변수튜닝)
- [12. 확장 방법](#12-확장-방법)
- [13. 문제 해결](#13-문제-해결)
- [14. 라이선스](#14-라이선스)

---

## 1. 빠른 시작

### 요구 사항

- Python **>= 3.10**
- Linux 환경 권장
- 인터넷 필요: WMT24++ 다운로드, 모델 체크포인트, COMET/MetricX 모델
- 대형 모델(120B/235B)은 NVIDIA GPU 다수 필요

> 일부 모델(Gemma 등)은 HF 라이선스 승인/로그인이 필요할 수 있습니다.

### 1) 의존성 설치

```bash
uv sync
```

선택: 개발 도구 포함

```bash
uv sync --extra dev
```

### 2) 환경 점검 (선택)

```bash
./scripts/doctor.sh
```

### 3) MetricX 설치 (MetricX 사용 시)

```bash
./scripts/fetch_metricx.sh
```

### 4) 데이터 준비

```bash
./scripts/prepare_data.sh wmt24pp en-ko_KR
```

### 5) vLLM 서버 실행 + 번역 생성

```bash
./scripts/serve_vllm.sh gpt_oss_120b 8000
./scripts/wait_server.sh http://localhost:8000/v1
./scripts/generate.sh run1 wmt24pp en-ko_KR gpt_oss_120b http://localhost:8000/v1
```

### 6) 점수화 + 집계

```bash
./scripts/score.sh run1 xcomet_mqm    wmt24pp en-ko_KR gpt_oss_120b
./scripts/score.sh run1 xcomet_qe     wmt24pp en-ko_KR gpt_oss_120b
./scripts/score.sh run1 metricx24_ref wmt24pp en-ko_KR gpt_oss_120b
./scripts/score.sh run1 metricx24_qe  wmt24pp en-ko_KR gpt_oss_120b
./scripts/aggregate.sh run1
```

요약 결과: `outputs/run1/summary.csv`

---

## 2. 결과물 개요

1) **준비된 데이터(JSONL)**  
`data/<dataset>/<lp>.jsonl`

2) **생성 결과(JSONL)**  
`outputs/<run>/gen/<dataset>/<lp>/<model_key>.jsonl`  
추가 필드: `hypothesis`, `model`, `served_model`, `gen_params`

3) **메트릭 결과(JSONL)**  
`outputs/<run>/metrics/<metric_key>/<dataset>/<lp>/<model_key>.jsonl`  
추가 필드: `metric`, `score`, (옵션) `error_spans`

4) **요약 CSV**  
`outputs/<run>/summary.csv`  
`mean/min/max/n` 등 통계 포함

### JSONL 예시

준비된 데이터:

```json
{"id":"en-ko_KR:123","lp":"en-ko_KR","domain":"news","source":"...","reference":"...","original_reference":"..."}
```

생성 결과:

```json
{"id":"en-ko_KR:123","lp":"en-ko_KR","source":"...","reference":"...","hypothesis":"...","model":"gpt_oss_120b","served_model":"gpt-oss-120b","gen_params":{"temperature":0.0,"top_p":1.0,"max_tokens":256,"stop":[]}}
```

메트릭 결과:

```json
{"id":"en-ko_KR:123","metric":"xcomet_mqm","score":0.123}
```

---

## 3. 레포 구조

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
│   └── metricx/            # scripts/fetch_metricx.sh로 생성
├── data/                   # 준비된 데이터 (gitignore)
└── outputs/                # 생성/점수 결과 (gitignore)
```

구성의 핵심은 `configs/`입니다.

---

## 4. 설치 (uv)

```bash
uv sync
```

CLI 직접 실행 예:

```bash
uv run evalmt-prepare --dataset wmt24pp --lps en-ko_KR --out data/wmt24pp
```

레포 루트 자동 탐색을 위해 `EVALMT_ROOT` 환경 변수를 사용할 수 있습니다.

---

## 5. vLLM 설치

환경에 따라 vLLM 설치 방식이 달라집니다.

### 5.1 일반 vLLM

```bash
uv pip install vllm
```

### 5.2 gpt-oss 전용 빌드

```bash
./scripts/install_vllm_gptoss.sh
```

`gpt-oss`는 전용 빌드가 필요할 수 있으니 실패 시 모델 카드/환경에 맞게 조정하세요.

---

## 6. MetricX 설치

MetricX는 `third_party/metricx` 아래에 클론됩니다.

```bash
./scripts/fetch_metricx.sh
```

MetricX 실행은 내부적으로 `python -m metricx24.predict`를 호출합니다.

---

## 7. 데이터 준비 (WMT24++)

WMT24++는 Hugging Face `google/wmt24pp`에서 내려받습니다.

```bash
./scripts/prepare_data.sh wmt24pp all
```

구성 키:
- `filter_bad_source`: 품질 낮은 소스 제거
- `use_post_edit_as_reference`: post-edit를 reference로 사용

`evalmt-prepare` CLI로 `--max-samples`와 `--seed` 샘플링도 가능합니다.

---

## 8. 파이프라인 실행

### 8.1 vLLM 서버 실행

```bash
./scripts/serve_vllm.sh gpt_oss_120b 8000
```

### 8.2 번역 생성

```bash
./scripts/generate.sh run1 wmt24pp en-ko_KR gpt_oss_120b http://localhost:8000/v1
```

- 기본 동시성: 16 (`CONCURRENCY`로 변경 가능)
- `--resume` 옵션으로 기존 결과를 건너뜀

### 8.3 메트릭 점수화

```bash
./scripts/score.sh run1 xcomet_mqm wmt24pp en-ko_KR gpt_oss_120b
```

### 8.4 집계

```bash
./scripts/aggregate.sh run1
```

### 8.5 한 번에 실행

```bash
./scripts/run_all.sh run1 wmt24pp all \
  gpt_oss_120b,qwen3_235b_a22b_instruct_2507 \
  xcomet_mqm,xcomet_qe
```

`run_all.sh`는 모델별로 **생성 후 vLLM 종료 → 점수화** 순서로 동작합니다.

---

## 9. 스크립트/CLI 요약

### 스크립트

- `scripts/uv_sync.sh`: `uv sync` 래퍼
- `scripts/doctor.sh`: 환경 점검
- `scripts/install_vllm_gptoss.sh`: gpt-oss용 vLLM 설치
- `scripts/fetch_metricx.sh`: MetricX 클론 + 의존성 설치
- `scripts/prepare_data.sh`: 데이터 준비
- `scripts/serve_vllm.sh`: vLLM 서버 실행
- `scripts/wait_server.sh`: 서버 준비 대기
- `scripts/generate.sh`: 번역 생성
- `scripts/score.sh`: 메트릭 점수화
- `scripts/aggregate.sh`: 요약 CSV 생성
- `scripts/run_all.sh`: 전체 파이프라인 실행

### CLI 엔트리포인트

- `evalmt-prepare`
- `evalmt-serve-vllm`
- `evalmt-wait-server`
- `evalmt-generate`
- `evalmt-score`
- `evalmt-aggregate`

---

## 10. 구성 시스템

모든 설정은 YAML로 정의됩니다.

### 10.1 데이터셋 (`configs/datasets/*.yaml`)

예: `configs/datasets/wmt24pp.yaml`

- `type`: 데이터셋 로더 유형
- `hf_repo`: HF repo ID
- `prepared_dir`: 출력 디렉터리

### 10.2 모델 (`configs/models/*.yaml`)

예: `configs/models/gpt_oss_120b.yaml`

- `hf_model_id`: vLLM이 서빙할 모델
- `served_model_name`: OpenAI 요청 시 모델명
- `vllm`: `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len` 등
- `prompt.system`, `prompt.user`: 프롬프트 템플릿
- `generation_defaults`: `temperature`, `top_p`, `max_tokens`

프롬프트에는 `{target_language}`, `{target_region}`, `{source}`가 주입됩니다.

### 10.3 메트릭 (`configs/metrics/*.yaml`)

예: `configs/metrics/xcomet_mqm.yaml`

- `type`: `comet` 또는 `metricx`
- `mode`: `ref` 또는 `qe`
- `direction`: `higher_is_better` / `lower_is_better`

주의: MetricX는 **낮을수록 좋음**이 기본입니다.

---

## 11. 환경 변수/튜닝

- `CONCURRENCY`: 생성 동시성 (`scripts/generate.sh`)
- `SCORE_GPU_LIST`: 점수화 전용 GPU 지정  
  예: `SCORE_GPU_LIST=7 ./scripts/score.sh ...`
- `CUDA_VISIBLE_DEVICES`: vLLM/Metric GPU 제한
- `HF_HOME`: HF 캐시 경로
- `HF_HUB_ENABLE_HF_TRANSFER=1`: 다운로드 가속
- `EVALMT_ROOT`: 레포 루트 강제 지정

### H100 x8 노드 팁

- 대형 모델은 `tensor_parallel_size: 8` 권장
- vLLM 서버와 메트릭 스코어링을 동시에 수행하면 OOM 위험
- 필요 시 서버를 중단한 뒤 스코어링 수행 (`run_all.sh` 방식)

---

## 12. 확장 방법

### 새 모델 추가

1) `configs/models/<model_key>.yaml` 추가  
2) `serve_vllm.sh` / `generate.sh`로 실행

### 새 메트릭 추가

1) `evalmt/metrics/`에 러너 추가  
2) `evalmt/metrics/registry.py`에 등록  
3) `configs/metrics/*.yaml` 작성

### 새 데이터셋 추가

1) `evalmt/datasets/`에 로더 구현  
2) 레지스트리 등록  
3) `configs/datasets/*.yaml` 추가

---

## 13. 문제 해결

### vLLM 서버 준비 지연

- `gpu_memory_utilization` / `max_model_len` 낮추기
- `tensor_parallel_size` 확인
- `gpt-oss` 전용 vLLM 설치 여부 확인

### MetricX 모듈 오류

- `third_party/metricx` 존재 확인
- `./scripts/fetch_metricx.sh` 재실행

### 출력 텍스트에 불필요한 래퍼 포함

- 정리 로직: `evalmt/generation/vllm_openai.py`
- 필요 시 접두어/패턴 추가

---

## 14. 라이선스

라이선스는 `LICENSE`를 참고하세요.
