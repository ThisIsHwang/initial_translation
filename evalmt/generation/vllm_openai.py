from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

import httpx


def _chat_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def clean_translation(text: str) -> str:
    """Best-effort cleanup to keep output as plain translation text."""

    t = (text or "").strip()

    # Remove code fences
    if t.startswith("```"):
        m = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", t, flags=re.S)
        if m:
            t = m.group(1).strip()

    # Common prefixes
    for prefix in [
        "Translation:",
        "translation:",
        "번역:",
        "번역문:",
        "译文：",
        "翻译：",
    ]:
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()

    # Strip surrounding quotes (minimal)
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    return t


async def chat_completion(
    *,
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[List[str]] = None,
    timeout_s: float = 120.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    url = _chat_endpoint(api_base)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    backoff = 1.5
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                return r.json()
        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt >= max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2.0


def extract_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return ""
