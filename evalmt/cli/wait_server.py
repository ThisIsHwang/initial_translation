from __future__ import annotations

import argparse
import asyncio

from ..utils.net import wait_for_openai_server


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", required=True)
    p.add_argument("--timeout", type=int, default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(wait_for_openai_server(args.api_base, timeout_s=args.timeout))
    print("✅ server ready:", args.api_base)


if __name__ == "__main__":
    main()
