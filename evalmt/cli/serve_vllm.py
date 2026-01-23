from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from typing import Any, Dict, List

from ..config import load_model_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model key in configs/models/*.yaml")
    p.add_argument("--port", type=int, default=None)
    return p.parse_args()


def _bool_flag(name: str, value: bool) -> List[str]:
    return [name] if value else []


def main() -> None:
    args = parse_args()
    cfg = load_model_config(args.model)

    hf_model_id = cfg["hf_model_id"]
    served = cfg.get("served_model_name", hf_model_id)
    vllm_cfg: Dict[str, Any] = cfg.get("vllm", {})

    port = args.port if args.port is not None else int(vllm_cfg.get("port", 8000))
    host = str(vllm_cfg.get("host", "0.0.0.0"))

    cmd: List[str] = ["vllm", "serve", hf_model_id]
    cmd += ["--host", host, "--port", str(port)]
    cmd += ["--served-model-name", served]

    if "dtype" in vllm_cfg:
        cmd += ["--dtype", str(vllm_cfg["dtype"])]
    if "tensor_parallel_size" in vllm_cfg:
        cmd += ["--tensor-parallel-size", str(vllm_cfg["tensor_parallel_size"])]
    if "gpu_memory_utilization" in vllm_cfg:
        cmd += ["--gpu-memory-utilization", str(vllm_cfg["gpu_memory_utilization"])]
    if "max_model_len" in vllm_cfg:
        cmd += ["--max-model-len", str(vllm_cfg["max_model_len"])]
    if "tokenizer" in vllm_cfg:
        cmd += ["--tokenizer", str(vllm_cfg["tokenizer"])]
    if "tokenizer_mode" in vllm_cfg:
        cmd += ["--tokenizer-mode", str(vllm_cfg["tokenizer_mode"])]
    if "hf_config_path" in vllm_cfg:
        cmd += ["--hf-config-path", str(vllm_cfg["hf_config_path"])]
    if "hf_overrides" in vllm_cfg:
        cmd += ["--hf-overrides", json.dumps(vllm_cfg["hf_overrides"])]
    if "chat_template" in vllm_cfg:
        cmd += ["--chat-template", str(vllm_cfg["chat_template"])]

    cmd += _bool_flag("--trust-remote-code", bool(vllm_cfg.get("trust_remote_code", False)))

    extra = vllm_cfg.get("extra_args", [])
    if extra:
        cmd += [str(x) for x in extra]

    print("Launching:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
