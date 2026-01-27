from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sacrebleu.metrics import BLEU

from ..utils.jsonl import iter_jsonl, write_jsonl
from .base import BaseMetric
from .registry import register_metric


@register_metric("bleu")
class BleuMetric(BaseMetric):
    @staticmethod
    def _extract_lang_from_lp(lp: str) -> str:
        if not lp or "-" not in lp:
            return ""
        tgt = lp.split("-")[-1]
        if "_" in tgt:
            tgt = tgt.split("_")[0]
        return tgt.lower()

    @staticmethod
    def _infer_target_lang(rows: List[Dict[str, Any]], field: Optional[str]) -> str:
        if field:
            for r in rows:
                v = r.get(field)
                if isinstance(v, str) and v.strip():
                    v = v.strip()
                    if field == "lp":
                        return BleuMetric._extract_lang_from_lp(v)
                    if "-" in v:
                        return v.split("-")[0].strip().lower()
                    if "_" in v:
                        v = v.split("_")[0]
                    return v.strip().lower()
            return ""

        for key in ("lp", "target_language_code", "tgt_lang", "target_lang"):
            for r in rows:
                v = r.get(key)
                if isinstance(v, str) and v.strip():
                    if key == "lp":
                        return BleuMetric._extract_lang_from_lp(v.strip())
                    if "_" in v:
                        v = v.split("_")[0]
                    return v.strip().lower()
        return ""

    @staticmethod
    def _select_tokenizer(target_lang: str, *, asian_support: bool) -> Optional[str]:
        if not target_lang:
            return None
        if target_lang.startswith("ko"):
            return "ko-mecab"
        if target_lang.startswith("zh"):
            return "zh" if asian_support else None
        return None

    def score(self, *, gen_path: Path, out_path: Path, tmp_dir: Path) -> None:
        mode = self.cfg.get("mode", "ref")
        if mode != "ref":
            raise ValueError("BLEU is reference-based only (mode=ref).")

        mt_field = str(self.cfg.get("mt_field", "hypothesis"))
        ref_field = str(self.cfg.get("ref_field", "reference"))

        case_sensitive = self.cfg.get("case_sensitive", None)
        if case_sensitive is None:
            lowercase = bool(self.cfg.get("lowercase", False))
        else:
            lowercase = not bool(case_sensitive)

        smooth_method = self.cfg.get("smooth_method", "exp")
        smooth_value = self.cfg.get("smooth_value", None)
        max_ngram_order = self.cfg.get("max_ngram_order", None)
        effective_order = bool(self.cfg.get("effective_order", True))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = list(iter_jsonl(gen_path))
        if not rows:
            raise ValueError(f"No rows to score in {gen_path}")

        if any(mt_field not in r for r in rows):
            raise KeyError(f"Missing '{mt_field}' field for BLEU input in {gen_path}")
        if any(ref_field not in r for r in rows):
            raise KeyError(f"Missing '{ref_field}' field for BLEU input in {gen_path}")

        tokenize = self.cfg.get("tokenize", None)
        if tokenize is None:
            tokenize = self.cfg.get("tokenizer", None)
        asian_support = bool(self.cfg.get("asian_support", False))
        target_lang = self._infer_target_lang(rows=rows, field=self.cfg.get("target_lang_field"))
        if not tokenize:
            tokenize = self._select_tokenizer(target_lang, asian_support=asian_support)

        bleu_kwargs: Dict[str, Any] = {
            "lowercase": lowercase,
            "tokenize": tokenize,
            "smooth_method": smooth_method,
            "smooth_value": smooth_value,
            "effective_order": effective_order,
        }
        if max_ngram_order is not None:
            bleu_kwargs["max_ngram_order"] = int(max_ngram_order)

        bleu = BLEU(**bleu_kwargs)

        hyps = [r.get(mt_field, "") or "" for r in rows]
        refs = [r.get(ref_field, "") or "" for r in rows]

        scored_rows: List[Dict[str, Any]] = []
        for r, hyp, ref in zip(rows, hyps, refs):
            rr = dict(r)
            rr["metric"] = self.metric_key
            rr["score"] = float(bleu.sentence_score(hyp, [ref]).score)
            scored_rows.append(rr)

        write_jsonl(out_path, scored_rows, append=False)

        sys_score = float(bleu.corpus_score(hyps, [refs]).score)
        (out_path.parent / f"{out_path.stem}.system_score.txt").write_text(str(sys_score), encoding="utf-8")
