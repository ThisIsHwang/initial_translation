from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseMetric(ABC):
    def __init__(self, metric_key: str, cfg: Dict[str, Any]) -> None:
        self.metric_key = metric_key
        self.cfg = cfg

    @abstractmethod
    def score(self, *, gen_path: Path, out_path: Path, tmp_dir: Path) -> None:
        raise NotImplementedError
