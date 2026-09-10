"""Loader for the multilingual SMS spam collection (22 language columns).

Source: dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset (machine-
translated SMS Spam Collection). Each row has labels (ham/spam), the English
text, and per-language translations. We emit one record per non-empty
language column so the model sees scam patterns in 20+ languages.

Language codes map to our spec tiers: hi, bn, mr, pa (Tier 1 India) plus
es, fr, de, pt, ar, ru, id, ur, zh, ja, ko, tr, jv, uk, sv, no.
"""

import csv
import io

from scamless.data.schemas import make_message

SOURCE_PREFIX = "multilingual_sms"
LANG_COLUMNS = {
    "text": "en",
    "text_hi": "hi",
    "text_de": "de",
    "text_fr": "fr",
    "text_es": "es",
    "text_zh": "zh",
    "text_ar": "ar",
    "text_bn": "bn",
    "text_ru": "ru",
    "text_pt": "pt",
    "text_ja": "ja",
    "text_id": "id",
    "text_ur": "ur",
    "text_pa": "pa",
    "text_jv": "jv",
    "text_tr": "tr",
    "text_ko": "ko",
    "text_mr": "mr",
    "text_uk": "uk",
    "text_sv": "sv",
    "text_no": "no",
}
MIN_TEXT_CHARS = 25  # SMS are short; keep the bar lower than email


def records_from_csv_bytes(raw: bytes) -> list[dict]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace")))
    records = []
    for row in reader:
        label = (row.get("labels") or "").strip().lower()
        labels = ["generic_spam"] if label == "spam" else []
        for column, lang in LANG_COLUMNS.items():
            text = (row.get(column) or "").strip()
            if len(text) < MIN_TEXT_CHARS:
                continue
            records.append(
                make_message(text, list(labels), f"{SOURCE_PREFIX}_{lang}", language=lang)
            )
    return records
