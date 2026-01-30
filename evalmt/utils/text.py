from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def infer_order_field(rows: List[Dict[str, Any]], order_field: Optional[str]) -> Optional[str]:
    if order_field:
        return order_field
    for cand in ("segment_id", "no", "idx"):
        if any(cand in r for r in rows):
            return cand
    return None


def join_with_sep(parts: List[str], sep: str, *, add_space: Optional[bool] = None) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if not sep:
        return " ".join(parts)
    if add_space is None:
        if not sep.isspace() and "\n" not in sep and "\t" not in sep and " " not in sep:
            glue = f" {sep} "
        else:
            glue = sep
    else:
        glue = f" {sep} " if add_space else sep
    return glue.join(parts)
