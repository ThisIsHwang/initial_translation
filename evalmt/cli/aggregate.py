from __future__ import annotations

import argparse

import pandas as pd

from ..config import ROOT, load_metric_config
from ..utils.jsonl import iter_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "outputs" / args.run
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"No metrics dir: {metrics_dir}")

    rows = []
    for metric_key_dir in metrics_dir.iterdir():
        if not metric_key_dir.is_dir():
            continue
        metric_key = metric_key_dir.name
        mcfg = load_metric_config(metric_key)
        direction = mcfg.get("direction", "unknown")

        for dataset_dir in metric_key_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for lp_dir in dataset_dir.iterdir():
                if not lp_dir.is_dir():
                    continue
                lp = lp_dir.name
                for f in lp_dir.glob("*.jsonl"):
                    model = f.stem
                    scores = []
                    for r in iter_jsonl(f):
                        if "score" in r:
                            scores.append(float(r["score"]))

                    if not scores:
                        continue

                    rows.append(
                        {
                            "run": args.run,
                            "dataset": dataset,
                            "lp": lp,
                            "model": model,
                            "metric": metric_key,
                            "direction": direction,
                            "n": len(scores),
                            "mean": sum(scores) / len(scores),
                            "min": min(scores),
                            "max": max(scores),
                        }
                    )

    df = pd.DataFrame(rows).sort_values(["dataset", "lp", "metric", "model"])
    out_csv = run_dir / "summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"✅ summary -> {out_csv}")


if __name__ == "__main__":
    main()
