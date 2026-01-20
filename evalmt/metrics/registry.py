from __future__ import annotations

from typing import Dict, Type

from .base import BaseMetric

METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}


def register_metric(metric_type: str):
    def _wrap(cls):
        METRIC_REGISTRY[metric_type] = cls
        return cls

    return _wrap
