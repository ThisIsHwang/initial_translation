from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils.jsonl import iter_jsonl, write_jsonl
from ..utils.text import infer_order_field, join_with_sep, normalize_text
from ..align.labse_align import AlignConfig, align_with_labse
from ..generation.vllm_openai import chat_completion, extract_text


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to recover JSON object/array embedded in extra text.
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    candidates: List[str] = []
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append(text[start_obj : end_obj + 1])
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidates.append(text[start_arr : end_arr + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    raise ValueError(f"Failed to parse JSON from alignment model response:\n{text}")


def _group_indices(
    rows: List[Dict[str, Any]],
    *,
    doc_field: str,
    order_field: Optional[str],
) -> Tuple[List[Any], Dict[Any, List[int]]]:
    has_doc_field = doc_field and any(r.get(doc_field) is not None for r in rows)
    order_field = infer_order_field(rows, order_field)

    groups: Dict[Any, List[int]] = {}
    order: List[Any] = []
    for i, r in enumerate(rows):
        if has_doc_field:
            doc_id = r.get(doc_field)
            if doc_id is None:
                doc_id = f"__missing_doc_{i}"
        else:
            doc_id = "__all__"
        if doc_id not in groups:
            groups[doc_id] = []
            order.append(doc_id)
        groups[doc_id].append(i)

    if order_field:
        for doc_id, idxs in groups.items():
            vals = [rows[i].get(order_field) for i in idxs]
            if all(v is not None for v in vals):
                groups[doc_id] = sorted(idxs, key=lambda i: rows[i].get(order_field))

    return order, groups


def _build_doc_rows(
    rows: List[Dict[str, Any]],
    *,
    fields: List[str],
    sep: str,
    marker_template: Optional[str],
    marker_join: str,
    marker_fields: List[str],
    doc_field: str,
    order_field: Optional[str],
    include_segment_ids: bool,
) -> List[Dict[str, Any]]:
    doc_order, groups = _group_indices(rows, doc_field=doc_field, order_field=order_field)
    out_rows: List[Dict[str, Any]] = []
    order_field = infer_order_field(rows, order_field)

    for doc_id in doc_order:
        idxs = groups[doc_id]
        base = dict(rows[idxs[0]])
        if doc_id is not None:
            base[doc_field] = doc_id
            base["id"] = f"doc:{doc_id}"

        for field in fields:
            parts = [normalize_text(rows[i].get(field)) for i in idxs]
            if marker_template and field in marker_fields:
                out = ""
                for k, part in enumerate(parts, start=1):
                    marker = marker_template.format(i=k)
                    if out:
                        out = f"{out}{marker_join}{marker}{marker_join}{part}"
                    else:
                        out = f"{marker}{marker_join}{part}" if marker_join else f"{marker}{part}"
                base[field] = out
            else:
                base[field] = join_with_sep(parts, sep)

        base["segment_count"] = len(idxs)
        if include_segment_ids:
            if order_field:
                base["segment_ids"] = [rows[i].get(order_field) for i in idxs]
            else:
                base["segment_ids"] = idxs
        out_rows.append(base)

    return out_rows


def _split_text(
    text: str,
    *,
    sep: str,
    splitter: str,
    regex: Optional[str],
    marker_regex: Optional[str],
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if splitter == "marker" and marker_regex:
        parts = [p.strip() for p in re.split(marker_regex, text) if p.strip()]
        return parts

    if splitter == "auto" and marker_regex and re.search(marker_regex, text):
        parts = [p.strip() for p in re.split(marker_regex, text) if p.strip()]
        return parts

    if splitter in ("auto", "sep") and sep and sep in text:
        parts = [p.strip() for p in text.split(sep)]
        return [p for p in parts if p]

    if splitter == "sep":
        return [text]

    pattern = regex or r"(?<=[.!?。！？])\s+"
    chunks = [c.strip() for c in text.replace("\r\n", "\n").split("\n") if c.strip()]
    parts: List[str] = []
    for c in chunks:
        segs = [s.strip() for s in re.split(pattern, c) if s.strip()]
        if len(segs) == 1:
            segs = [s.strip() for s in re.split(r"(?<=[.!?。！？])", c) if s.strip()]
        parts.extend(segs)
    return [p for p in parts if p]


def _align_segments(parts: List[str], target_n: int, *, marker_regex: Optional[str]) -> Tuple[List[str], str]:
    if marker_regex:
        parts = [re.sub(marker_regex, "", p).strip() for p in parts]
    if target_n <= 0:
        return [], "empty"
    if len(parts) == target_n:
        return parts, "exact"
    if len(parts) > target_n:
        merged = parts[: target_n - 1] + [" ".join(parts[target_n - 1 :]).strip()]
        return merged, "merged"
    padded = parts + [""] * (target_n - len(parts))
    return padded, "padded"


def _parse_fields(rows: List[Dict[str, Any]], fields_arg: Optional[str]) -> List[str]:
    if fields_arg:
        fields = [f.strip() for f in fields_arg.split(",") if f.strip()]
        return fields
    fields = ["source"]
    if any("reference" in r for r in rows):
        fields.append("reference")
    if any("hypothesis" in r for r in rows):
        fields.append("hypothesis")
    return fields


def cmd_to_doc(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = list(iter_jsonl(in_path))
    if not rows:
        raise ValueError(f"No rows in {in_path}")

    fields = _parse_fields(rows, args.fields)
    # Drop fields that don't exist in any row (QE datasets may omit reference).
    fields_present = []
    for f in fields:
        if any(f in r for r in rows):
            fields_present.append(f)
        else:
            print(f"⚠️  Skipping missing field '{f}' in {in_path}")
    fields = fields_present

    marker_fields = [x.strip() for x in (args.marker_fields or "").split(",") if x.strip()]

    doc_rows = _build_doc_rows(
        rows,
        fields=fields,
        sep=args.sep,
        marker_template=args.marker_template,
        marker_join=args.marker_join,
        marker_fields=marker_fields,
        doc_field=args.doc_field,
        order_field=args.order_field,
        include_segment_ids=args.include_segment_ids,
    )

    write_jsonl(out_path, doc_rows, append=False)
    print(f"✅ doc jsonl -> {out_path} (docs={len(doc_rows)})")


def cmd_expand(args: argparse.Namespace) -> None:
    base_path = Path(args.base)
    doc_path = Path(args.doc)
    out_path = Path(args.output)

    base_rows = list(iter_jsonl(base_path))
    if not base_rows:
        raise ValueError(f"No rows in {base_path}")

    doc_rows = list(iter_jsonl(doc_path))
    if not doc_rows:
        raise ValueError(f"No rows in {doc_path}")

        doc_field = args.doc_field
        order_field = infer_order_field(base_rows, args.order_field)

    doc_has_field = doc_field and any(r.get(doc_field) is not None for r in doc_rows)
    doc_map: Dict[Any, Dict[str, Any]] = {}
    if doc_has_field:
        for r in doc_rows:
            doc_id = r.get(doc_field)
            if doc_id is not None:
                doc_map[doc_id] = r

    doc_order, groups = _group_indices(base_rows, doc_field=doc_field, order_field=order_field)

    sent_hyps: List[str] = ["" for _ in base_rows]
    doc_split_status: List[str] = ["" for _ in base_rows]
    doc_hyps: List[str] = ["" for _ in base_rows]
    align_scores: List[Optional[float]] = [None for _ in base_rows]
    align_spans: List[Optional[Tuple[int, int]]] = [None for _ in base_rows]
    align_low_conf: List[Optional[bool]] = [None for _ in base_rows]

    for doc_idx, doc_id in enumerate(doc_order):
        idxs = groups[doc_id]
        if doc_has_field:
            doc_row = doc_map.get(doc_id, {})
        else:
            doc_row = doc_rows[doc_idx] if doc_idx < len(doc_rows) else {}

        doc_hyp = normalize_text(doc_row.get(args.hyp_field))
        if args.align_mode == "gpt":
            if not args.align_api_base or not args.align_model_name:
                raise ValueError("align_mode=gpt requires --align-api-base and --align-model-name")

            src_sents = [base_rows[i].get("source", "") for i in idxs]

            system = (
                "You are a sentence alignment engine. "
                "Return JSON only, with schema: {\"aligned\":[{\"src\":...,\"hyp\":...}]} . "
                "Given src_sents (N items) and hyp_text, split hyp_text into N chunks in order. "
                "Each output item must have keys: src, hyp. "
                "Do NOT change src text. "
                "Keep monotonic order. "
                "If you cannot find content, use empty string for hyp."
            )
            user = json.dumps({"src_sents": src_sents, "hyp_text": doc_hyp}, ensure_ascii=False)

            schema = {
                "type": "object",
                "properties": {
                    "aligned": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {"type": "string"},
                                "hyp": {"type": "string"},
                            },
                            "required": ["src", "hyp"],
                        },
                    }
                },
                "required": ["aligned"],
            }

            response_format = None
            if args.align_response_format == "json_schema":
                response_format = {"type": "json_schema", "json_schema": {"name": "alignment", "schema": schema}}

            async def _run(fmt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return await chat_completion(
                    api_base=args.align_api_base,
                    model=args.align_model_name,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=args.align_temperature,
                    top_p=1.0,
                    max_tokens=args.align_max_tokens,
                    response_format=fmt,
                )

            try:
                resp = asyncio.run(_run(response_format))
            except Exception:
                resp = asyncio.run(_run(None))

            text = extract_text(resp)
            try:
                data = _safe_json_loads(text)
            except Exception:
                if response_format:
                    resp = asyncio.run(_run(None))
                    text = extract_text(resp)
                    data = _safe_json_loads(text)
                else:
                    raise

            items = data.get("aligned") if isinstance(data, dict) else data
            if not isinstance(items, list) or len(items) != len(src_sents):
                raise ValueError("Alignment output size mismatch")

            for i, idx in enumerate(idxs):
                row = items[i] if isinstance(items[i], dict) else {}
                sent_hyps[idx] = (row.get("hyp") if row else "") or ""
                doc_split_status[idx] = "gpt"
                doc_hyps[idx] = doc_hyp
        elif args.align_mode == "labse":
            src_sents = [base_rows[i].get("source", "") for i in idxs]
            attach_remaining = bool(args.align_attach_remaining_to_last)
            if args.align_no_attach_remaining:
                attach_remaining = False

            cfg = AlignConfig(
                model_name=args.align_model,
                device=args.align_device or None,
                batch_size=args.align_batch_size,
                seed=args.align_seed,
                max_chars=args.align_max_chars,
                min_tokens=args.align_min_tokens,
                min_chars=args.align_min_chars,
                max_merge_chunks=args.align_max_merge_chunks,
                max_merge_chars=args.align_max_merge_chars,
                attach_remaining_to_last=attach_remaining,
                allow_n_to_1=args.align_allow_n_to_1,
                low_conf_threshold=args.align_low_conf_threshold,
            )
            aligned_rows, _doc_score = align_with_labse(src_sents, doc_hyp, config=cfg)
            for i, idx in enumerate(idxs):
                row = aligned_rows[i] if i < len(aligned_rows) else None
                sent_hyps[idx] = (row.get("hyp") if row else "") or ""
                doc_split_status[idx] = "labse"
                doc_hyps[idx] = doc_hyp
                if args.align_meta and row:
                    align_scores[idx] = float(row.get("score", 0.0))
                    align_spans[idx] = tuple(row.get("hyp_span", (0, -1)))
                    align_low_conf[idx] = bool(row.get("low_conf", False))
        else:
            parts = _split_text(
                doc_hyp,
                sep=args.sep,
                splitter=args.splitter,
                regex=args.regex,
                marker_regex=args.marker_regex,
            )
            aligned, status = _align_segments(parts, len(idxs), marker_regex=args.marker_regex)

            for i, idx in enumerate(idxs):
                sent_hyps[idx] = aligned[i] if i < len(aligned) else ""
                doc_split_status[idx] = status
                doc_hyps[idx] = doc_hyp

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(base_rows):
        rr = dict(r)
        rr[args.hyp_field] = sent_hyps[i]
        if args.add_doc_hyp:
            rr["doc_hypothesis"] = doc_hyps[i]
            rr["doc_split_status"] = doc_split_status[i]
        if args.align_meta:
            if align_scores[i] is not None:
                rr["align_score"] = align_scores[i]
            if align_spans[i] is not None:
                rr["align_span"] = align_spans[i]
            if align_low_conf[i] is not None:
                rr["align_low_conf"] = align_low_conf[i]
        out_rows.append(rr)

    write_jsonl(out_path, out_rows, append=False)
    print(f"✅ expanded jsonl -> {out_path} (rows={len(out_rows)})")


def cmd_clean(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = list(iter_jsonl(in_path))
    if not rows:
        raise ValueError(f"No rows in {in_path}")

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    rx = re.compile(args.marker_regex) if args.marker_regex else None

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        for f in fields:
            if f in rr and isinstance(rr[f], str) and rx:
                rr[f] = re.sub(rx, "", rr[f]).strip()
        out_rows.append(rr)

    write_jsonl(out_path, out_rows, append=False)
    print(f"✅ cleaned jsonl -> {out_path} (rows={len(out_rows)})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("to-doc")
    p_doc.add_argument("--input", required=True)
    p_doc.add_argument("--output", required=True)
    p_doc.add_argument("--sep", default="</s>")
    p_doc.add_argument("--doc-field", default="document_id")
    p_doc.add_argument("--order-field", default=None)
    p_doc.add_argument("--fields", default=None, help="comma-separated fields to concat")
    p_doc.add_argument("--include-segment-ids", action="store_true")
    p_doc.add_argument("--marker-template", default=None, help="e.g., ⟦{i}⟧")
    p_doc.add_argument("--marker-join", default=" ")
    p_doc.add_argument("--marker-fields", default=None, help="fields to insert markers into")

    p_exp = sub.add_parser("expand")
    p_exp.add_argument("--base", required=True, help="sentence-level base jsonl")
    p_exp.add_argument("--doc", required=True, help="doc-level jsonl with hypothesis")
    p_exp.add_argument("--output", required=True)
    p_exp.add_argument("--sep", default="</s>")
    p_exp.add_argument("--doc-field", default="document_id")
    p_exp.add_argument("--order-field", default=None)
    p_exp.add_argument("--hyp-field", default="hypothesis")
    p_exp.add_argument("--splitter", choices=["auto", "sep", "regex", "marker"], default="auto")
    p_exp.add_argument("--regex", default=None)
    p_exp.add_argument("--marker-regex", default=None)
    p_exp.add_argument("--add-doc-hyp", action="store_true")
    p_exp.add_argument("--align-mode", choices=["rule", "labse", "gpt"], default="rule")
    p_exp.add_argument("--align-meta", action="store_true")
    p_exp.add_argument("--align-model", default="sentence-transformers/LaBSE")
    p_exp.add_argument("--align-device", default=None)
    p_exp.add_argument("--align-batch-size", type=int, default=32)
    p_exp.add_argument("--align-seed", type=int, default=42)
    p_exp.add_argument("--align-max-chars", type=int, default=320)
    p_exp.add_argument("--align-min-tokens", type=int, default=4)
    p_exp.add_argument("--align-min-chars", type=int, default=12)
    p_exp.add_argument("--align-max-merge-chunks", type=int, default=6)
    p_exp.add_argument("--align-max-merge-chars", type=int, default=1000)
    p_exp.add_argument("--align-attach-remaining-to-last", action="store_true")
    p_exp.add_argument("--align-no-attach-remaining", action="store_true")
    p_exp.add_argument("--align-allow-n-to-1", action="store_true")
    p_exp.add_argument("--align-low-conf-threshold", type=float, default=0.55)
    p_exp.add_argument("--align-api-base", default=None)
    p_exp.add_argument("--align-model-name", default=None)
    p_exp.add_argument("--align-temperature", type=float, default=0.0)
    p_exp.add_argument("--align-max-tokens", type=int, default=64000)
    p_exp.add_argument("--align-response-format", choices=["none", "json_schema"], default="none")

    p_clean = sub.add_parser("clean")
    p_clean.add_argument("--input", required=True)
    p_clean.add_argument("--output", required=True)
    p_clean.add_argument("--marker-regex", default=r"⟦\d+⟧")
    p_clean.add_argument("--fields", default="source,reference,hypothesis")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "to-doc":
        cmd_to_doc(args)
    elif args.cmd == "expand":
        cmd_expand(args)
    elif args.cmd == "clean":
        cmd_clean(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
