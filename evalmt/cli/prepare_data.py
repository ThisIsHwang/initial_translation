from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_dataset_config, load_lang_code_map
from ..datasets.registry import DATASET_REGISTRY
from ..datasets.wmt24pp import WMT24PPDataset, discover_wmt24pp_lps  # register side-effect
from ..utils.jsonl import iter_jsonl, write_jsonl
from ..utils.lang_codes import apply_lang_code_map, split_lp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset key (ex: wmt24pp)")
    p.add_argument("--lps", required=True, help="all | comma-separated list (ex: en-ko_KR,en-ja_JP)")
    p.add_argument("--out", required=True, help="output dir for prepared jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lang-code-map",
        default=None,
        help="JSON string or path to JSON mapping for source/target codes",
    )
    return p.parse_args()


def _load_lang_code_map(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("lang_code_map must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def _postprocess_lang_codes(
    *,
    lps: list[str],
    out_dir: Path,
    lang_code_map: dict[str, str] | None,
) -> None:
    for lp in lps:
        path = out_dir / f"{lp}.jsonl"
        if not path.exists():
            continue
        rows = []
        for r in iter_jsonl(path):
            row_lp = (r.get("lp") or lp or "").strip()
            src_code = (r.get("source_lang_code") or "").strip()
            tgt_code = (r.get("target_lang_code") or "").strip()
            if not src_code or not tgt_code:
                guess_src, guess_tgt = split_lp(row_lp)
                src_code = src_code or guess_src
                tgt_code = tgt_code or guess_tgt
            src_code = apply_lang_code_map(src_code, lang_code_map)
            tgt_code = apply_lang_code_map(tgt_code, lang_code_map)
            r["source_lang_code"] = src_code
            r["target_lang_code"] = tgt_code
            rows.append(r)
        write_jsonl(path, rows, append=False)


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.dataset)
    ds_type = cfg["type"]

    if ds_type not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset type: {ds_type}. Registered={list(DATASET_REGISTRY)}")

    if args.lps == "all":
        if ds_type == "wmt24pp":
            lps = discover_wmt24pp_lps(cfg["hf_repo"])
        else:
            raise SystemExit("lps=all is only implemented for wmt24pp in this repo.")
    else:
        lps = [x.strip() for x in args.lps.split(",") if x.strip()]

    out_dir = Path(args.out)
    cli_map = _load_lang_code_map(args.lang_code_map)
    config_map = load_lang_code_map()
    if config_map and not isinstance(config_map, dict):
        raise ValueError("configs/lang_codes.yaml must be a dict")
    lang_code_map = cli_map or config_map or None

    ds = DATASET_REGISTRY[ds_type](
        hf_repo=cfg["hf_repo"],
        repo_type=cfg.get("repo_type", "dataset"),
        filter_bad_source=bool(cfg.get("filter_bad_source", True)),
        use_post_edit_as_reference=bool(cfg.get("use_post_edit_as_reference", True)),
    )
    ds.prepare(
        lps=lps,
        out_dir=out_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        lang_code_map=lang_code_map,
    )
    _postprocess_lang_codes(lps=lps, out_dir=out_dir, lang_code_map=lang_code_map)


if __name__ == "__main__":
    main()
