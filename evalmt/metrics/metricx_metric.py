from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..config import ROOT
from ..utils.jsonl import iter_jsonl, write_jsonl
from .base import BaseMetric
from .registry import register_metric


@register_metric("metricx")
class MetricXMetric(BaseMetric):
    def score(self, *, gen_path: Path, out_path: Path, tmp_dir: Path) -> None:
        variant = self.cfg.get("variant", "metricx24")
        mode = self.cfg.get("mode", "ref")  # ref | qe
        tokenizer = self.cfg["tokenizer"]
        model_name_or_path = self.cfg["model_name_or_path"]
        max_input_length = int(self.cfg.get("max_input_length", 1536))
        batch_size = int(self.cfg.get("batch_size", 1))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        in_jsonl = tmp_dir / f"{out_path.stem}.metricx_input.jsonl"
        gen_rows = list(iter_jsonl(gen_path))

        metricx_rows: List[Dict[str, Any]] = []
        for r in gen_rows:
            ref = "" if mode == "qe" else r.get("reference", "")
            metricx_rows.append({
                "source": r["source"],
                "hypothesis": r["hypothesis"],
                "reference": ref,
            })
        write_jsonl(in_jsonl, metricx_rows, append=False)

        pred_jsonl = tmp_dir / f"{out_path.stem}.metricx_pred.jsonl"

        metricx_repo = ROOT / "third_party" / "metricx"
        if not metricx_repo.exists():
            raise FileNotFoundError(
                f"MetricX repo not found at {metricx_repo}. Run ./scripts/fetch_metricx.sh"
            )

        env = dict(os.environ)
        env["PYTHONPATH"] = str(metricx_repo) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        module = "metricx24.predict" if variant == "metricx24" else "metricx23.predict"
        cmd = [
            "python",
            "-m",
            module,
            "--tokenizer",
            tokenizer,
            "--model_name_or_path",
            model_name_or_path,
            "--max_input_length",
            str(max_input_length),
            "--batch_size",
            str(batch_size),
            "--input_file",
            str(in_jsonl),
            "--output_file",
            str(pred_jsonl),
        ]
        if mode == "qe":
            cmd.append("--qe")

        subprocess.run(cmd, env=env, check=True)

        pred_rows = list(iter_jsonl(pred_jsonl))
        if len(pred_rows) != len(gen_rows):
            raise RuntimeError(f"MetricX output size mismatch: {len(pred_rows)} vs {len(gen_rows)}")

        merged: List[Dict[str, Any]] = []
        for r, p in zip(gen_rows, pred_rows):
            rr = dict(r)
            rr["metric"] = self.metric_key
            rr["score"] = float(p.get("prediction"))
            merged.append(rr)

        write_jsonl(out_path, merged, append=False)
