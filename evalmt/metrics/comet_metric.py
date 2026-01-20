from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from comet import download_model, load_from_checkpoint

from ..utils.jsonl import iter_jsonl, write_jsonl
from .base import BaseMetric
from .registry import register_metric


@register_metric("comet")
class CometMetric(BaseMetric):
    def score(self, *, gen_path: Path, out_path: Path, tmp_dir: Path) -> None:
        model_id = self.cfg["model"]
        mode = self.cfg.get("mode", "ref")  # ref | qe
        batch_size = int(self.cfg.get("batch_size", 8))
        gpus = int(self.cfg.get("gpus", 1))
        export_spans = bool(self.cfg.get("export_error_spans", False))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        model_path = download_model(model_id)
        model = load_from_checkpoint(model_path)

        rows = list(iter_jsonl(gen_path))
        comet_in: List[Dict[str, Any]] = []
        for r in rows:
            if mode == "qe":
                comet_in.append({"src": r["source"], "mt": r["hypothesis"]})
            else:
                comet_in.append({"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]})

        out = model.predict(comet_in, batch_size=batch_size, gpus=gpus)

        spans = None
        if export_spans:
            spans = getattr(getattr(out, "metadata", None), "error_spans", None)

        scored_rows: List[Dict[str, Any]] = []
        for i, r in enumerate(rows):
            rr = dict(r)
            rr["metric"] = self.metric_key
            rr["score"] = float(out.scores[i])
            if spans is not None:
                try:
                    rr["error_spans"] = spans[i]
                except Exception:
                    rr["error_spans"] = None
            scored_rows.append(rr)

        write_jsonl(out_path, scored_rows, append=False)

        sys_score = getattr(out, "system_score", None)
        if sys_score is not None:
            (out_path.parent / f"{out_path.stem}.system_score.txt").write_text(str(sys_score), encoding="utf-8")
