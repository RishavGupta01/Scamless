"""Loader for the Hugging Face seven-phishing-email-datasets parquet corpus.

203,017 emails (108,689 benign / 94,328 phishing-or-spam) aggregated from
TREC-05/06/07, CEAS-08, Enron, Ling, and SpamAssassin. Label semantics:
0 = benign, 1 = phishing/spam (the source label buckets them together, so
rows land as phishing OR generic_spam depending on content markers).

The corpus is downloaded as parquet shards by the fetch step (network lives
in download.py); this module maps rows to our MessageRecord shape.
"""

import pandas as pd

from scamless.data.download import RAW
from scamless.data.schemas import make_message
from scamless.data.spamassassin import enrich_labels

SOURCE = "hf_seven_phishing"
HF_ID = "puyang2025/seven-phishing-email-datasets"
MIN_TEXT_CHARS = 40

_SHARDS = [f"train-0000{i}-of-00008.parquet" for i in range(8)]


def fetch_shards(session) -> list:
    """Download the 8 parquet shards (idempotent, atomic via fetch)."""
    from scamless.data.download import fetch

    paths = []
    for shard in _SHARDS:
        url = f"https://huggingface.co/datasets/{HF_ID}/resolve/main/{shard}"
        dest = RAW / "seven_phishing" / shard
        paths.append(fetch(url, dest, session))
    return paths


def records_from_dataframe(df: pd.DataFrame) -> list[dict]:
    texts = df["text"].astype("string").fillna("") if "text" in df.columns else pd.Series(dtype="string")
    labels = df["label"].astype("Int64").fillna(0) if "label" in df.columns else pd.Series(dtype="Int64")
    records = []
    for text, label in zip(texts, labels):
        text = str(text).strip()
        if len(text) < MIN_TEXT_CHARS:
            continue
        if int(label) == 1:
            labels_out = enrich_labels(text, ["generic_spam"])
        else:
            labels_out = []
        records.append(make_message(text, labels_out, SOURCE, language="en"))
    return records


def collect() -> list[dict]:
    all_records = []
    seven_dir = RAW / "seven_phishing"
    for shard in sorted(seven_dir.glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["text", "label"])
        all_records.extend(records_from_dataframe(df))
    return all_records
