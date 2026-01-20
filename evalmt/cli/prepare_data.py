from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_dataset_config
from ..datasets.registry import DATASET_REGISTRY
from ..datasets.wmt24pp import WMT24PPDataset, discover_wmt24pp_lps  # register side-effect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset key (ex: wmt24pp)")
    p.add_argument("--lps", required=True, help="all | comma-separated list (ex: en-ko_KR,en-ja_JP)")
    p.add_argument("--out", required=True, help="output dir for prepared jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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

    ds = DATASET_REGISTRY[ds_type](
        hf_repo=cfg["hf_repo"],
        repo_type=cfg.get("repo_type", "dataset"),
        filter_bad_source=bool(cfg.get("filter_bad_source", True)),
        use_post_edit_as_reference=bool(cfg.get("use_post_edit_as_reference", True)),
    )
    ds.prepare(lps=lps, out_dir=out_dir, max_samples=args.max_samples, seed=args.seed)


if __name__ == "__main__":
    main()
