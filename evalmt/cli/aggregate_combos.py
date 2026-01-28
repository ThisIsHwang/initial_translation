from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--doc-suffix", default="_doc")
    p.add_argument("--out", default=None)
    return p.parse_args()


def infer_combo(dataset: str, model: str, metric: str, *, doc_suffix: str) -> tuple[str, str, str]:
    # Eval level is driven by context metrics (ctx => doc-level scoring context).
    eval_level = "doc" if metric.endswith("_ctx") else "sent"

    if model.endswith("__from_doc"):
        gen_level = "doc"
    elif model.endswith("__from_sent"):
        gen_level = "sent"
    else:
        gen_level = "sent"

    # If dataset is an explicit doc dataset, override eval_level to doc.
    if dataset.endswith(doc_suffix):
        eval_level = "doc"

    combo = f"{gen_level}->{eval_level}"
    return gen_level, eval_level, combo


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "outputs" / args.run
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise SystemExit("summary.csv is empty")

    gen_levels = []
    eval_levels = []
    combos = []
    for _, row in df.iterrows():
        gen_level, eval_level, combo = infer_combo(
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("metric", "")),
            doc_suffix=args.doc_suffix,
        )
        gen_levels.append(gen_level)
        eval_levels.append(eval_level)
        combos.append(combo)

    df.insert(len(df.columns), "gen_level", gen_levels)
    df.insert(len(df.columns), "eval_level", eval_levels)
    df.insert(len(df.columns), "combo", combos)

    out_path = Path(args.out) if args.out else run_dir / "summary_combos.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ combos summary -> {out_path}")


if __name__ == "__main__":
    main()
