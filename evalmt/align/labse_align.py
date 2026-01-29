"""
LaBSE-based monotonic sentence alignment
======================================

Usage example
-------------
from evalmt.align.labse_align import align_with_labse

src_sents = ["Hello world.", "This is a test."]
hyp_text = "Bonjour le monde. Ceci est un test."

aligned, doc_score = align_with_labse(src_sents, hyp_text)
for row in aligned:
    print(row["hyp"], row["score"], row["hyp_span"], row["low_conf"])

Notes
-----
- This module aligns a list of already-split source sentences to a hypothesis
  paragraph by splitting the hypothesis into micro-chunks and performing a
  greedy monotonic 1–N alignment using LaBSE embeddings.
- It is language-agnostic (no language detection) and supports multilingual
  scripts by relying on Unicode punctuation and character-count heuristics.
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class AlignConfig:
    model_name: str = "sentence-transformers/LaBSE"
    backend: Optional[str] = None  # "labse" | "e5"
    device: Optional[str] = None
    batch_size: int = 32
    seed: int = 42

    # Micro-chunk rules
    max_chars: int = 320
    min_tokens: int = 4
    min_chars: int = 12

    # Alignment rules
    max_merge_chunks: int = 6
    max_merge_chars: int = 1000
    attach_remaining_to_last: bool = True
    allow_n_to_1: bool = False

    # Confidence
    low_conf_threshold: float = 0.55
    worst_k: int = 3

    # E5 settings
    e5_query_prefix: str = "query: "
    e5_passage_prefix: str = "passage: "
    e5_max_length: int = 512


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_PUNCT_SET = r"\.\?\!;:。！？؛۔॥؟።፧፨ฯ…"
_SPLIT_REGEX = re.compile(rf"([\\n{_PUNCT_SET}]+)")


def _basic_split(text: str) -> List[str]:
    if not text:
        return []
    parts: List[str] = []
    buf = ""
    for piece in _SPLIT_REGEX.split(text):
        if not piece:
            continue
        if _SPLIT_REGEX.fullmatch(piece):
            # attach delimiter to previous buffer
            buf += piece
            continue
        if buf:
            parts.append(buf.strip())
            buf = ""
        buf = piece
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _force_split_long(chunks: List[str], max_chars: int) -> List[str]:
    out: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
            continue
        start = 0
        while start < len(c):
            out.append(c[start : start + max_chars])
            start += max_chars
    return out


def _merge_short(chunks: List[str], min_tokens: int, min_chars: int) -> List[str]:
    if not chunks:
        return []
    out: List[str] = []
    buf = ""

    def _flush():
        nonlocal buf
        if buf:
            out.append(buf)
            buf = ""

    for c in chunks:
        if not buf:
            buf = c
        else:
            buf = f"{buf} {c}"

        tok_count = len(buf.split())
        if tok_count >= min_tokens or len(buf) >= min_chars:
            _flush()

    _flush()
    return out


def split_to_micro_chunks(
    hyp_text: str,
    *,
    max_chars: int,
    min_tokens: int,
    min_chars: int,
) -> List[str]:
    if not hyp_text:
        return []

    chunks = _basic_split(hyp_text)
    chunks = _force_split_long(chunks, max_chars)
    chunks = _merge_short(chunks, min_tokens, min_chars)
    return chunks


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _encode_texts(
    model: Any,
    texts: Iterable[str],
    *,
    batch_size: int,
    device: str,
    tokenizer: Optional[Any] = None,
    max_length: int = 512,
) -> np.ndarray:
    if isinstance(model, SentenceTransformer):
        vecs = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        return vecs.astype(np.float32, copy=False)

    if tokenizer is None:
        raise ValueError("tokenizer is required for transformer backend")

    all_vecs: List[np.ndarray] = []
    texts_list = list(texts)
    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i : i + batch_size]
        batch_dict = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        last_hidden = outputs.last_hidden_state
        mask = batch_dict["attention_mask"]
        pooled = _average_pool(last_hidden, mask)
        pooled = F.normalize(pooled, p=2, dim=1)
        all_vecs.append(pooled.detach().cpu().numpy())

    vecs = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 1), dtype=np.float32)
    return vecs.astype(np.float32, copy=False)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized so dot == cosine
    return float(np.dot(a, b))


def align_with_labse(
    src_sents: List[str],
    hyp_text: str,
    *,
    ref_text: Optional[str] = None,
    config: Optional[AlignConfig] = None,
    low_conf_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    cfg = config or AlignConfig()
    set_seed(cfg.seed)

    device = cfg.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.allow_n_to_1:
        LOGGER.warning("allow_n_to_1=True is not implemented; proceeding with 1–N only.")

    backend = cfg.backend
    if not backend:
        if "intfloat/multilingual-e5" in cfg.model_name:
            backend = "e5"
        else:
            backend = "labse"

    tokenizer = None
    if backend == "e5":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModel.from_pretrained(cfg.model_name).to(device)
        model.eval()
    else:
        model = SentenceTransformer(cfg.model_name, device=device)

    hyp_chunks = split_to_micro_chunks(
        hyp_text,
        max_chars=cfg.max_chars,
        min_tokens=cfg.min_tokens,
        min_chars=cfg.min_chars,
    )
    ref_chunks: Optional[List[str]] = None
    if ref_text is not None:
        ref_chunks = split_to_micro_chunks(
            ref_text,
            max_chars=cfg.max_chars,
            min_tokens=cfg.min_tokens,
            min_chars=cfg.min_chars,
        )

    if not src_sents:
        return [], {"mean": 0.0, "worst_k_mean": 0.0}

    if backend == "e5":
        src_texts = [f"{cfg.e5_query_prefix}{s}" for s in src_sents]
    else:
        src_texts = src_sents
    src_vecs = _encode_texts(
        model,
        src_texts,
        batch_size=cfg.batch_size,
        device=device,
        tokenizer=tokenizer,
        max_length=cfg.e5_max_length,
    )

    aligned: List[Dict[str, Any]] = []
    j = 0
    hyp_len = len(hyp_chunks)

    for i, src in enumerate(src_sents):
        if j >= hyp_len:
            aligned.append(
                {
                    "src": src,
                    "hyp": "",
                    "score": 0.0,
                    "hyp_span": (j, j - 1),
                    "low_conf": True,
                    "debug": {"best_j": j - 1, "merged_chunks": 0},
                }
            )
            continue

        max_k = min(hyp_len, j + cfg.max_merge_chunks)
        candidates: List[str] = []
        cur = ""
        cur_chars = 0
        for k in range(j, max_k):
            part = hyp_chunks[k]
            if cur:
                cur = f"{cur} {part}"
            else:
                cur = part
            cur_chars = len(cur)
            if cur_chars > cfg.max_merge_chars:
                break
            candidates.append(cur)

        if backend == "e5":
            cand_texts = [f"{cfg.e5_passage_prefix}{c}" for c in candidates]
        else:
            cand_texts = candidates
        cand_vecs = _encode_texts(
            model,
            cand_texts,
            batch_size=cfg.batch_size,
            device=device,
            tokenizer=tokenizer,
            max_length=cfg.e5_max_length,
        )

        best_score = -math.inf
        best_k = j - 1
        for offset, cand_vec in enumerate(cand_vecs):
            score = _cos_sim(src_vecs[i], cand_vec)
            if score > best_score:
                best_score = score
                best_k = j + offset

        if best_k < j:
            aligned.append(
                {
                    "src": src,
                    "hyp": "",
                    "score": 0.0,
                    "hyp_span": (j, j - 1),
                    "low_conf": True,
                    "debug": {"best_j": j - 1, "merged_chunks": 0},
                }
            )
            continue

        hyp_piece = " ".join(hyp_chunks[j : best_k + 1])

        # Attach remaining chunks to last sentence if requested
        if i == len(src_sents) - 1 and cfg.attach_remaining_to_last and best_k < hyp_len - 1:
            hyp_piece = " ".join(hyp_chunks[j:])
            best_k = hyp_len - 1
            if backend == "e5":
                hyp_texts = [f"{cfg.e5_passage_prefix}{hyp_piece}"]
            else:
                hyp_texts = [hyp_piece]
            best_score = _cos_sim(
                src_vecs[i],
                _encode_texts(
                    model,
                    hyp_texts,
                    batch_size=cfg.batch_size,
                    device=device,
                    tokenizer=tokenizer,
                    max_length=cfg.e5_max_length,
                )[0],
            )

        low_conf = best_score < cfg.low_conf_threshold
        row: Dict[str, Any] = {
            "src": src,
            "hyp": hyp_piece,
            "score": float(best_score),
            "hyp_span": (j, best_k),
            "low_conf": bool(low_conf),
            "debug": {"best_j": int(best_k), "merged_chunks": int(best_k - j + 1)},
        }
        if ref_chunks is not None:
            if best_k < len(ref_chunks):
                row["ref"] = " ".join(ref_chunks[j : best_k + 1])
            else:
                row["ref"] = ""
        aligned.append(row)

        if low_conf and low_conf_hook:
            try:
                low_conf_hook(row)
            except Exception:
                LOGGER.exception("low_conf_hook failed")

        j = best_k + 1

    scores = [float(r.get("score", 0.0)) for r in aligned]
    mean_score = float(np.mean(scores)) if scores else 0.0
    k = max(1, min(cfg.worst_k, len(scores)))
    worst_k_mean = float(np.mean(sorted(scores)[:k])) if scores else 0.0
    doc_score = {"mean": mean_score, "worst_k_mean": worst_k_mean}

    return aligned, doc_score
