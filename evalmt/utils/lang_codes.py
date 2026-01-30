from __future__ import annotations


def split_lp(lp: str) -> tuple[str, str]:
    """Split an lp like 'en-ko_KR' into (src, tgt)."""
    if not lp or "-" not in lp:
        return "", ""
    src, tgt = lp.rsplit("-", 1)
    return src, tgt


def apply_lang_code_map(code: str, lang_code_map: dict[str, str] | None) -> str:
    if not lang_code_map or not code:
        return code
    if code in lang_code_map and lang_code_map[code]:
        return lang_code_map[code]
    alt = code.replace("_", "-")
    if alt in lang_code_map and lang_code_map[alt]:
        return lang_code_map[alt]
    alt2 = code.replace("-", "_")
    if alt2 in lang_code_map and lang_code_map[alt2]:
        return lang_code_map[alt2]
    base = code.split("-", 1)[0].split("_", 1)[0]
    if base in lang_code_map and lang_code_map[base]:
        return lang_code_map[base]
    lower = code.lower()
    if lower in lang_code_map and lang_code_map[lower]:
        return lang_code_map[lower]
    return code
