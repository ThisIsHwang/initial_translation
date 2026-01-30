from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from comet import download_model, load_from_checkpoint

from ..utils.jsonl import iter_jsonl, write_jsonl
from ..utils.text import infer_order_field, join_with_sep, normalize_text
from .base import BaseMetric
from .registry import register_metric


@register_metric("comet")
class CometMetric(BaseMetric):
    def _build_context_fields(
        self,
        rows: List[Dict[str, Any]],
        *,
        window: int,
        sep: str,
        sep_with_spaces: bool,
        append_current: bool,
        append_delim: str,
        append_only_if_context: bool,
        doc_field: str,
        order_field: Optional[str],
        src_field: str,
        mt_field: str,
        ref_field: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        n = len(rows)
        ctx_src = [""] * n
        ctx_mt = [""] * n
        ctx_ref = [""] * n

        def _maybe_append(seq: str, cur: str, has_context: bool) -> str:
            if not append_current or not cur:
                return seq
            if append_only_if_context and not has_context:
                return seq
            delim = append_delim if append_delim is not None else "\n"
            return f"{seq}{delim}{cur}" if seq else cur

        if window <= 0:
            for i, r in enumerate(rows):
                cur_src = normalize_text(r.get(src_field))
                cur_mt = normalize_text(r.get(mt_field))
                cur_ref = normalize_text(r.get(ref_field))
                ctx_src[i] = _maybe_append(cur_src, cur_src, False)
                ctx_mt[i] = _maybe_append(cur_mt, cur_mt, False)
                ctx_ref[i] = _maybe_append(cur_ref, cur_ref, False)
            return ctx_src, ctx_mt, ctx_ref

        has_doc_field = doc_field and any(r.get(doc_field) is not None for r in rows)
        order_field = infer_order_field(rows, order_field)

        doc_groups: Dict[Any, List[int]] = {}
        for i, r in enumerate(rows):
            if has_doc_field:
                doc_id = r.get(doc_field)
                if doc_id is None:
                    doc_id = f"__missing_doc_{i}"
            else:
                doc_id = "__all__"
            doc_groups.setdefault(doc_id, []).append(i)

        for _, idxs in doc_groups.items():
            idxs_sorted = idxs
            if order_field:
                order_vals = [rows[i].get(order_field) for i in idxs]
                if all(v is not None for v in order_vals):
                    idxs_sorted = sorted(idxs, key=lambda i: rows[i].get(order_field))

            for pos, idx in enumerate(idxs_sorted):
                start = max(0, pos - window)
                ctx_idxs = idxs_sorted[start:pos]

                src_parts = [normalize_text(rows[j].get(src_field)) for j in ctx_idxs]
                mt_parts = [normalize_text(rows[j].get(mt_field)) for j in ctx_idxs]
                ref_parts = [normalize_text(rows[j].get(ref_field)) for j in ctx_idxs]

                src_parts.append(normalize_text(rows[idx].get(src_field)))
                mt_parts.append(normalize_text(rows[idx].get(mt_field)))
                ref_parts.append(normalize_text(rows[idx].get(ref_field)))

                has_context = len(ctx_idxs) > 0
                cur_src = normalize_text(rows[idx].get(src_field))
                cur_mt = normalize_text(rows[idx].get(mt_field))
                cur_ref = normalize_text(rows[idx].get(ref_field))

                src_seq = join_with_sep(src_parts, sep, add_space=sep_with_spaces)
                mt_seq = join_with_sep(mt_parts, sep, add_space=sep_with_spaces)
                ref_seq = join_with_sep(ref_parts, sep, add_space=sep_with_spaces)

                ctx_src[idx] = _maybe_append(src_seq, cur_src, has_context)
                ctx_mt[idx] = _maybe_append(mt_seq, cur_mt, has_context)
                ctx_ref[idx] = _maybe_append(ref_seq, cur_ref, has_context)

        return ctx_src, ctx_mt, ctx_ref

    def score(self, *, gen_path: Path, out_path: Path, tmp_dir: Path) -> None:
        model_id = self.cfg["model"]
        mode = self.cfg.get("mode", "ref")  # ref | qe
        batch_size = int(self.cfg.get("batch_size", 8))
        gpus = int(self.cfg.get("gpus", 1))
        export_spans = bool(self.cfg.get("export_error_spans", False))
        enable_context = bool(self.cfg.get("enable_context", False))
        context_window = int(self.cfg.get("context_window", 0))
        context_sep = str(self.cfg.get("context_separator", "</s>"))
        context_sep_with_spaces = bool(self.cfg.get("context_separator_with_spaces", True))
        context_append_current = bool(self.cfg.get("context_append_current", False))
        context_append_delim = str(self.cfg.get("context_append_delimiter", "\n"))
        context_append_only_if_context = bool(self.cfg.get("context_append_only_if_context", False))
        context_doc_field = str(self.cfg.get("context_doc_field", "document_id"))
        context_order_field = self.cfg.get("context_order_field", None)
        src_field = str(self.cfg.get("src_field", "source"))
        mt_field = str(self.cfg.get("mt_field", "hypothesis"))
        ref_field = str(self.cfg.get("ref_field", "reference"))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        model_path = download_model(model_id)
        model = load_from_checkpoint(model_path)

        rows = list(iter_jsonl(gen_path))
        if not rows:
            raise ValueError(f"No rows to score in {gen_path}")

        if any(src_field not in r for r in rows):
            raise KeyError(f"Missing '{src_field}' field for COMET input in {gen_path}")
        if any(mt_field not in r for r in rows):
            raise KeyError(f"Missing '{mt_field}' field for COMET input in {gen_path}")
        if mode != "qe" and any(ref_field not in r for r in rows):
            raise KeyError(f"Missing '{ref_field}' field for COMET input in {gen_path}")

        if context_window > 0:
            enable_context = True

        ctx_src, ctx_mt, ctx_ref = self._build_context_fields(
            rows,
            window=context_window,
            sep=context_sep,
            sep_with_spaces=context_sep_with_spaces,
            append_current=context_append_current,
            append_delim=context_append_delim,
            append_only_if_context=context_append_only_if_context,
            doc_field=context_doc_field,
            order_field=context_order_field,
            src_field=src_field,
            mt_field=mt_field,
            ref_field=ref_field,
        )

        comet_in: List[Dict[str, Any]] = []
        for i, _r in enumerate(rows):
            if mode == "qe":
                comet_in.append({"src": ctx_src[i], "mt": ctx_mt[i]})
            else:
                comet_in.append({"src": ctx_src[i], "mt": ctx_mt[i], "ref": ctx_ref[i]})

        try:
            out = model.predict(comet_in, batch_size=batch_size, gpus=gpus, enable_context=enable_context)
        except TypeError:
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
