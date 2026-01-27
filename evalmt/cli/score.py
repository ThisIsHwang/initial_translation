from __future__ import annotations

import argparse

from ..config import ROOT, load_metric_config
from ..metrics.registry import METRIC_REGISTRY
from ..metrics.bleu_metric import BleuMetric  # register side-effect
from ..metrics.comet_metric import CometMetric  # register side-effect
from ..metrics.metricx_metric import MetricXMetric  # register side-effect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--lp", required=True)
    p.add_argument("--model", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_metric_config(args.metric)
    metric_type = cfg["type"]

    if metric_type not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric type: {metric_type}. Registered={list(METRIC_REGISTRY)}")

    gen_path = ROOT / "outputs" / args.run / "gen" / args.dataset / args.lp / f"{args.model}.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"Generation not found: {gen_path}")

    out_path = (
        ROOT / "outputs" / args.run / "metrics" / args.metric / args.dataset / args.lp / f"{args.model}.jsonl"
    )
    tmp_dir = ROOT / "outputs" / args.run / "tmp" / args.metric / args.dataset / args.lp / args.model

    metric = METRIC_REGISTRY[metric_type](args.metric, cfg)
    metric.score(gen_path=gen_path, out_path=out_path, tmp_dir=tmp_dir)

    print(f"✅ scored -> {out_path}")


if __name__ == "__main__":
    main()
