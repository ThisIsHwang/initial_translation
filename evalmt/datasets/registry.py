from __future__ import annotations

from typing import Dict, Type

from .base import BaseDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    def _wrap(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return _wrap
