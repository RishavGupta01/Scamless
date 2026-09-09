"""Loader for the Hugging Face ealvaradob/phishing-dataset 'texts' config.

Rows are {"text": str, "label": int} with 1 = phishing, 0 = benign. This fills
the phishing-label gap: Nazario has no stable direct-download mirror, while this
corpus provides ~7.7k phishing emails/SMS plus ~12.5k benign counterparts.
The raw texts.json is fetched by the downloader (no datasets-library script).
"""

import json

from scamless.data.schemas import make_message

SOURCE = "hf_phishing_texts"
SOURCE_HF_ID = "ealvaradob/phishing-dataset"
MIN_TEXT_CHARS = 40
LABEL_FOR_PHISHING = 1


def records_from_rows(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_TEXT_CHARS:
            continue
        labels = ["phishing"] if row.get("label") == LABEL_FOR_PHISHING else []
        records.append(make_message(text, labels, SOURCE, language="en"))
    return records


def records_from_json(raw: bytes) -> list[dict]:
    return records_from_rows(json.loads(raw.decode("utf-8")))
