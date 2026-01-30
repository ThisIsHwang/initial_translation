from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root.

    Priority:
    1) EVALMT_ROOT env var
    2) Walk up from current working directory until we see pyproject.toml and configs/
    3) Fallback to package directory parent

    This makes the tool usable even if installed, as long as you run it from the repo.
    """

    env = os.environ.get("EVALMT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    start = (start or Path.cwd()).resolve()
    cur = start
    for _ in range(10):
        if (cur / "pyproject.toml").exists() and (cur / "configs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # fallback: package parent
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root()
CONFIG_DIR = ROOT / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_config(dataset_key: str) -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "datasets" / f"{dataset_key}.yaml")


def load_model_config(model_key: str) -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "models" / f"{model_key}.yaml")


def load_metric_config(metric_key: str) -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "metrics" / f"{metric_key}.yaml")


def load_lang_code_map() -> Dict[str, Any]:
    path = CONFIG_DIR / "lang_codes.yaml"
    if not path.exists():
        return {}
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("configs/lang_codes.yaml must be a mapping")
    return data


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
