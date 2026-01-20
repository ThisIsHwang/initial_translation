from __future__ import annotations

import asyncio

import httpx


async def wait_for_openai_server(api_base: str, timeout_s: int = 600) -> None:
    """Wait for vLLM OpenAI-compatible server to become ready."""

    base = api_base.rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"

    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except httpx.RequestError:
                pass

            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Server not ready within {timeout_s}s: {url}")

            await asyncio.sleep(2.0)
