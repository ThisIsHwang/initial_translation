from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import list_repo_files, snapshot_download

from ..utils.jsonl import iter_jsonl, write_jsonl
from ..utils.lang_codes import split_lp
from .base import BaseDataset
from .registry import register_dataset


def discover_wmt24pp_lps(hf_repo: str) -> list[str]:
    """Discover language-pair files in google/wmt24pp.

    WMT24++ stores each LP as a jsonl file like: en-ko_KR.jsonl

    We list repo files via HF Hub metadata and extract filenames.
    """

    files = list_repo_files(repo_id=hf_repo, repo_type="dataset")
    lps = []
    for f in files:
        if f.startswith("en-") and f.endswith(".jsonl"):
            lps.append(Path(f).stem)
    # stable order
    return sorted(set(lps))


@register_dataset("wmt24pp")
class WMT24PPDataset(BaseDataset):
    def __init__(
        self,
        *,
        hf_repo: str,
        repo_type: str = "dataset",
        filter_bad_source: bool = True,
        use_post_edit_as_reference: bool = True,
    ) -> None:
        self.hf_repo = hf_repo
        self.repo_type = repo_type
        self.filter_bad_source = filter_bad_source
        self.use_post_edit_as_reference = use_post_edit_as_reference

    def _download(self, *, lps: list[str]) -> Path:
        allow_patterns = [f"{lp}.jsonl" for lp in lps] + ["README.md"]
        local_dir = snapshot_download(
            repo_id=self.hf_repo,
            repo_type=self.repo_type,
            allow_patterns=allow_patterns,
        )
        return Path(local_dir)

    def prepare(
        self,
        *,
        lps: list[str],
        out_dir: Path,
        max_samples: Optional[int] = None,
        seed: int = 42,
        lang_code_map: Optional[dict[str, str]] = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = self._download(lps=lps)

        rng = random.Random(seed)

        for lp in lps:
            src_path = repo_dir / f"{lp}.jsonl"
            if not src_path.exists():
                raise FileNotFoundError(f"Missing {src_path} (download failed?)")

            rows: list[Dict[str, Any]] = []
            for rec in iter_jsonl(src_path):
                if self.filter_bad_source and bool(rec.get("is_bad_source", False)):
                    continue

                # WMT24++ commonly uses `target` as post-edit and `original_target` as original MT.
                # We keep this flexible if keys differ.
                post_edit = rec.get("target") or rec.get("reference")
                original = rec.get("original_target") or rec.get("original_reference")

                ref = post_edit if self.use_post_edit_as_reference else (original or post_edit)
                row_lp = rec.get("lp", lp)
                src_code = rec.get("source_lang_code") or ""
                tgt_code = rec.get("target_lang_code") or ""
                if not src_code or not tgt_code:
                    guess_src, guess_tgt = split_lp(row_lp)
                    src_code = src_code or guess_src
                    tgt_code = tgt_code or guess_tgt

                rows.append(
                    {
                        "id": f"{lp}:{rec.get('segment_id', len(rows))}",
                        "lp": row_lp,
                        "domain": rec.get("domain"),
                        "document_id": rec.get("document_id"),
                        "segment_id": rec.get("segment_id"),
                        "source": rec.get("source"),
                        "reference": ref,
                        "original_reference": original,
                        "source_lang_code": src_code,
                        "target_lang_code": tgt_code,
                    }
                )

            if max_samples is not None and max_samples < len(rows):
                rng.shuffle(rows)
                rows = rows[:max_samples]

            out_path = out_dir / f"{lp}.jsonl"
            write_jsonl(out_path, rows, append=False)
            print(f"[wmt24pp] wrote {len(rows)} rows -> {out_path}")
