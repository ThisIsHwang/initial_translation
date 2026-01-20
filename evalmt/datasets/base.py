from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseDataset(ABC):
    @abstractmethod
    def prepare(
        self,
        *,
        lps: list[str],
        out_dir: Path,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        raise NotImplementedError
