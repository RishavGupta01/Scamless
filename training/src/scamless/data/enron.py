"""Loader for the Hugging Face SetFit/enron_spam dataset (network used in build only)."""

from scamless.data.schemas import make_message

SOURCE = "enron_spam"
SOURCE_HF_ID = "SetFit/enron_spam"
MIN_TEXT_CHARS = 40
LABEL_FOR_SPAM = 0  # SetFit/enron_spam: 0 = spam, 1 = ham


def records_from_hf_rows(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_TEXT_CHARS:
            continue
        labels = ["generic_spam"] if row.get("label") == LABEL_FOR_SPAM else []
        records.append(make_message(text, labels, SOURCE, language="en"))
    return records
