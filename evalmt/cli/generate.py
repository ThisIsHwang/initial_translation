from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

from ..config import ROOT, ensure_dir, load_dataset_config, load_model_config
from ..generation.prompts import (
    language_name_from_code,
    region_name_from_code,
    split_lang_pair,
)
from ..generation.vllm_openai import chat_completion, clean_translation, extract_text
from ..utils.jsonl import iter_jsonl
from ..utils.lang_codes import apply_lang_code_map


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


UNIFIED_SYSTEM_TEMPLATE = (
    "You are a professional {source_lang} ({src_lang_code}) to {target_lang}\n"
    "({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and\n"
    "nuances of the original {source_lang} text while adhering to {target_lang} grammar,\n"
    "vocabulary, and cultural sensitivities.\n"
    "Produce only the {target_lang}\n"
    "translation, without any additional explanations or commentary."
)
UNIFIED_USER_TEMPLATE = (
    "Please translate the following {source_lang} text into {target_lang}:\n\n\n{text}"
)


def _format_prompt(
    fmt: Dict[str, Any],
    *,
    prompt_style: str,
    sys_tmpl: str,
    usr_tmpl: str,
    merge_system: bool,
) -> tuple[str, str]:
    if prompt_style == "custom":
        system = sys_tmpl.format(**fmt) if sys_tmpl else ""
        user = usr_tmpl.format(**fmt) if usr_tmpl else ""
    else:
        system = UNIFIED_SYSTEM_TEMPLATE.format(**fmt)
        user = UNIFIED_USER_TEMPLATE.format(**fmt)
    if merge_system and system.strip():
        user = f"{system}\n{user}"
        system = ""
    return system, user


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
    prompt_style = str(model_cfg.get("prompt_style", "unified")).lower()
    hf_id = str(model_cfg.get("hf_model_id", "")).lower()
    model_key = str(args.model).lower()
    is_gemma = "gemma" in hf_id or "gemma" in model_key
    no_system_prompt = bool(model_cfg.get("no_system_prompt", is_gemma))
    src_code, tgt_code = ("", "")
    lang_code_map: Dict[str, str] = {}
    if message_format == "translategemma":
        src_code, tgt_code = split_lang_pair(args.lp)
        lang_code_map = model_cfg.get("translategemma_lang_code_map", {}) or {}
        if not isinstance(lang_code_map, dict):
            raise ValueError("translategemma_lang_code_map must be a dict in model config")
        env_map = os.environ.get("TRANSLATEGEMMA_LANG_CODE_MAP")
        if env_map:
            try:
                env_data = json.loads(env_map)
            except json.JSONDecodeError as exc:
                raise ValueError("TRANSLATEGEMMA_LANG_CODE_MAP must be valid JSON") from exc
            if isinstance(env_data, dict):
                lang_code_map = {**lang_code_map, **env_data}
            else:
                raise ValueError("TRANSLATEGEMMA_LANG_CODE_MAP must be a JSON object")

    gen_defaults = model_cfg.get("generation_defaults", {})
    temperature = float(gen_defaults.get("temperature", 0.0))
    top_p = float(gen_defaults.get("top_p", 1.0))
    max_tokens = int(gen_defaults.get("max_tokens", 256))
    stop = gen_defaults.get("stop", None)

    rows = [r for r in iter_jsonl(in_path) if r.get("id") not in done_ids]

    sem = asyncio.Semaphore(args.concurrency)

    def _row_lang_codes(row: Dict[str, Any]) -> tuple[str, str]:
        row_src = (row.get("source_lang_code") or "").strip()
        row_tgt = (row.get("target_lang_code") or "").strip()
        if row_src and row_tgt:
            return row_src, row_tgt
        lp_val = (row.get("lp") or args.lp or "").strip()
        if lp_val and "-" in lp_val:
            return split_lang_pair(lp_val)
        return row_src or src_code, row_tgt or tgt_code

    async def run_one(r: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            row_src, row_tgt = _row_lang_codes(r)
            src_lang = language_name_from_code(row_src)
            tgt_lang = language_name_from_code(row_tgt)
            tgt_region = region_name_from_code(row_tgt)
            fmt = {
                "source": r["source"],
                "text": r["source"],
                "source_lang": src_lang,
                "src_lang_code": row_src,
                "target_lang": tgt_lang,
                "tgt_lang_code": row_tgt,
                "target_language": tgt_lang,
                "target_region": tgt_region,
            }
            if message_format == "translategemma":
                if lang_code_map:
                    row_src = apply_lang_code_map(row_src, lang_code_map)
                    row_tgt = apply_lang_code_map(row_tgt, lang_code_map)
                tagged = f"<<<source>>>{row_src}<<<target>>>{row_tgt}<<<text>>>{r['source']}"
                messages = [{"role": "user", "content": tagged}]
            else:
                system, user = _format_prompt(
                    fmt,
                    prompt_style=prompt_style,
                    sys_tmpl=sys_tmpl,
                    usr_tmpl=usr_tmpl,
                    merge_system=no_system_prompt,
                )
                messages = []
                if system.strip():
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
                    "messages": messages,
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
