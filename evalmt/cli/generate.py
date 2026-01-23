from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

from ..config import ROOT, ensure_dir, load_dataset_config, load_model_config
from ..generation.prompts import (
    build_translategemma_messages,
    split_lang_pair,
    target_from_lp,
)
from ..generation.vllm_openai import chat_completion, clean_translation, extract_text
from ..utils.jsonl import iter_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--lp", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def out_gen_path(run: str, dataset: str, lp: str, model: str) -> Path:
    return ROOT / "outputs" / run / "gen" / dataset / lp / f"{model}.jsonl"


async def main_async() -> None:
    args = parse_args()
    ds_cfg = load_dataset_config(args.dataset)
    model_cfg = load_model_config(args.model)

    prepared_dir = Path(ds_cfg.get("prepared_dir", f"data/{args.dataset}"))
    in_path = prepared_dir / f"{args.lp}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {in_path}")

    out_path = out_gen_path(args.run, args.dataset, args.lp, args.model)
    ensure_dir(out_path.parent)

    done_ids: Set[str] = set()
    if args.resume and out_path.exists():
        for r in iter_jsonl(out_path):
            done_ids.add(r.get("id"))

    served = model_cfg.get("served_model_name", model_cfg["hf_model_id"])
    prompt_cfg = model_cfg.get("prompt", {})
    message_format = str(model_cfg.get("message_format", "simple")).lower()
    message_content_type = str(model_cfg.get("message_content_type", "text")).lower()
    sys_tmpl = prompt_cfg.get("system") or ""
    usr_tmpl = prompt_cfg.get("user") or "{source}"
    tgt = target_from_lp(args.lp)
    src_code, tgt_code = ("", "")
    if message_format == "translategemma":
        src_code, tgt_code = split_lang_pair(args.lp)

    gen_defaults = model_cfg.get("generation_defaults", {})
    temperature = float(gen_defaults.get("temperature", 0.0))
    top_p = float(gen_defaults.get("top_p", 1.0))
    max_tokens = int(gen_defaults.get("max_tokens", 256))
    stop = gen_defaults.get("stop", None)

    rows = [r for r in iter_jsonl(in_path) if r.get("id") not in done_ids]

    sem = asyncio.Semaphore(args.concurrency)

    async def run_one(r: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            if message_format == "translategemma":
                messages = build_translategemma_messages(
                    source_text=r["source"],
                    source_lang_code=src_code,
                    target_lang_code=tgt_code,
                    content_type=message_content_type,
                )
            else:
                system = sys_tmpl.format(target_language=tgt.language, target_region=tgt.region)
                user = usr_tmpl.format(
                    source=r["source"],
                    target_language=tgt.language,
                    target_region=tgt.region,
                )
                messages = []
                if sys_tmpl and system.strip():
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user})
            resp = await chat_completion(
                api_base=args.api_base,
                model=served,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop if stop else None,
            )
            text = clean_translation(extract_text(resp))

            out = dict(r)
            out.update(
                {
                    "model": args.model,
                    "served_model": served,
                    "hypothesis": text,
                    "gen_params": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "stop": stop,
                    },
                }
            )
            return out

    pending: Set[asyncio.Task] = set()
    window = args.concurrency * 4

    with out_path.open("a", encoding="utf-8") as f_out:
        pbar = tqdm(total=len(rows), desc=f"gen {args.model} {args.lp}")

        for r in rows:
            t = asyncio.create_task(run_one(r))
            pending.add(t)
            if len(pending) >= window:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    out_rec = d.result()
                    f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    pbar.update(1)

        for t in asyncio.as_completed(list(pending)):
            out_rec = await t
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            pbar.update(1)

        pbar.close()

    print(f"✅ wrote generations -> {out_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
