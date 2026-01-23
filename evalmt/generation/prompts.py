from __future__ import annotations

from dataclasses import dataclass

# Language / region mapping for WMT24++ target codes.
# Used only to render nicer prompts.

LANGUAGE_BY_CODE = {
    "ar_EG": "Arabic",
    "ar_SA": "Arabic",
    "bg_BG": "Bulgarian",
    "bn_IN": "Bengali",
    "ca_ES": "Catalan",
    "cs_CZ": "Czech",
    "da_DK": "Danish",
    "de_DE": "German",
    "deu": "German",
    "el_GR": "Greek",
    "es_MX": "Spanish",
    "et_EE": "Estonian",
    "fa_IR": "Farsi",
    "fi_FI": "Finnish",
    "fil_PH": "Filipino",
    "fra": "French",
    "fr_CA": "French",
    "fr_FR": "French",
    "gu_IN": "Gujarati",
    "he_IL": "Hebrew",
    "hi_IN": "Hindi",
    "hr_HR": "Croatian",
    "hu_HU": "Hungarian",
    "id_ID": "Indonesian",
    "is_IS": "Icelandic",
    "it_IT": "Italian",
    "ita": "Italian",
    "ja_JP": "Japanese",
    "jpn": "Japanese",
    "kn_IN": "Kannada",
    "ko_KR": "Korean",
    "kor": "Korean",
    "lt_LT": "Lithuanian",
    "lv_LV": "Latvian",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "nl_NL": "Dutch",
    "no_NO": "Norwegian",
    "pa_IN": "Punjabi",
    "pl_PL": "Polish",
    "pol": "Polish",
    "pt_BR": "Portuguese",
    "pt_PT": "Portuguese",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "rus": "Russian",
    "sk_SK": "Slovak",
    "sl_SI": "Slovenian",
    "spa": "Spanish",
    "sr_RS": "Serbian",
    "sv_SE": "Swedish",
    "sw_KE": "Swahili",
    "sw_TZ": "Swahili",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tr_TR": "Turkish",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "vi_VN": "Vietnamese",
    "vie": "Vietnamese",
    "zh_CN": "Mandarin",
    "zh_TW": "Mandarin",
    "zho": "Chinese",
    "zu_ZA": "Zulu",
}

REGION_BY_CODE = {
    "ar_EG": "Egypt",
    "ar_SA": "Saudi Arabia",
    "bg_BG": "Bulgaria",
    "bn_IN": "India",
    "ca_ES": "Spain",
    "cs_CZ": "Czechia",
    "da_DK": "Denmark",
    "de_DE": "Germany",
    "deu": "Germany",
    "el_GR": "Greece",
    "es_MX": "Mexico",
    "et_EE": "Estonia",
    "fa_IR": "Iran",
    "fi_FI": "Finland",
    "fil_PH": "Philippines",
    "fra": "France",
    "fr_CA": "Canada",
    "fr_FR": "France",
    "gu_IN": "India",
    "he_IL": "Israel",
    "hi_IN": "India",
    "hr_HR": "Croatia",
    "hu_HU": "Hungary",
    "id_ID": "Indonesia",
    "is_IS": "Iceland",
    "it_IT": "Italy",
    "ita": "Italy",
    "ja_JP": "Japan",
    "jpn": "Japan",
    "kn_IN": "India",
    "ko_KR": "South Korea",
    "kor": "South Korea",
    "lt_LT": "Lithuania",
    "lv_LV": "Latvia",
    "ml_IN": "India",
    "mr_IN": "India",
    "nl_NL": "Netherlands",
    "no_NO": "Norway",
    "pa_IN": "India",
    "pl_PL": "Poland",
    "pol": "Poland",
    "pt_BR": "Brazil",
    "pt_PT": "Portugal",
    "ro_RO": "Romania",
    "ru_RU": "Russia",
    "rus": "Russia",
    "sk_SK": "Slovakia",
    "sl_SI": "Slovenia",
    "spa": "Spain",
    "sr_RS": "Serbia",
    "sv_SE": "Sweden",
    "sw_KE": "Kenya",
    "sw_TZ": "Tanzania",
    "ta_IN": "India",
    "te_IN": "India",
    "th_TH": "Thailand",
    "tr_TR": "Turkey",
    "uk_UA": "Ukraine",
    "ur_PK": "Pakistan",
    "vi_VN": "Vietnam",
    "vie": "Vietnam",
    "zh_CN": "China",
    "zh_TW": "Taiwan",
    "zho": "China",
    "zu_ZA": "South Africa",
}
LANGUAGE_BY_CODE = {**REGION_BY_CODE, **LANGUAGE_BY_CODE}


@dataclass(frozen=True)
class TargetLang:
    code: str
    language: str
    region: str


def target_from_lp(lp: str) -> TargetLang:
    """"en-ko_KR" or "eng-US-deu" -> target code + friendly language/region."""

    if "-" not in lp:
        raise ValueError(f"Invalid language pair format: {lp}")
    tgt = lp.rsplit("-", 1)[1]
    return TargetLang(
        code=tgt,
        language=LANGUAGE_BY_CODE.get(tgt, tgt),
        region=REGION_BY_CODE.get(tgt, tgt),
    )

# TranslateGemma message schema helpers.

TRANSLATEGEMMA_LANGUAGE_BY_CODE = LANGUAGE_BY_CODE
TRANSLATEGEMMA_CODE_ALIASES = {}


def split_lang_pair(lp: str) -> tuple[str, str]:
    if "-" not in lp:
        raise ValueError(f"Invalid language pair format: {lp}")
    src, tgt = lp.rsplit("-", 1)
    return src, tgt


def _normalize_translategemma_lang_code(code: str) -> str:
    norm = (code or "").replace("-", "_")
    if "_" in norm:
        base, rest = norm.split("_", 1)
        base = TRANSLATEGEMMA_CODE_ALIASES.get(base, base)
        norm = f"{base}_{rest}"
    else:
        norm = TRANSLATEGEMMA_CODE_ALIASES.get(norm, norm)
    if norm in TRANSLATEGEMMA_LANGUAGE_BY_CODE:
        return norm
    return norm


def build_translategemma_messages(
    *,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
    content_type: str = "text",
) -> list[dict[str, object]]:
    src_code = _normalize_translategemma_lang_code(source_lang_code)
    tgt_code = _normalize_translategemma_lang_code(target_lang_code)
    if content_type == "image":
        content = {
            "type": "image",
            "source_lang_code": src_code,
            "target_lang_code": tgt_code,
            "image": (source_text or "").strip(),
        }
    elif content_type == "text":
        content = {
            "type": "text",
            "source_lang_code": src_code,
            "target_lang_code": tgt_code,
            "text": (source_text or "").strip(),
        }
    else:
        raise ValueError(f"Unsupported content_type: {content_type}")

    return [
        {
            "role": "user",
            "content": [content],
        }
    ]
