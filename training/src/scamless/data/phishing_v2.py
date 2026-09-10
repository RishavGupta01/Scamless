"""Loader for cybersectony/PhishingEmailDetectionv2.0 (URL-heavy corpus).

200,000 rows: 177,356 URLs + 22,644 emails, 4-class labels:
  0 = legitimate_email, 1 = phishing_email, 2 = legitimate_url, 3 = phishing_url

URLs feed the URL-model dataset (url records); emails feed the message
dataset. This gives the message model exposure to URL-shaped text too (raw
links inside messages) and stages a large supervised URL corpus for the
Phase 2 URL classifier.
"""

import pathlib

import pandas as pd

from scamless.data.download import RAW
from scamless.data.schemas import make_message, make_url

SOURCE = "hf_phishing_v2"
HF_ID = "cybersectony/PhishingEmailDetectionv2.0"
MIN_TEXT_CHARS = 20  # URLs are short by nature

_LABEL_MAP = {0: "legitimate_email", 1: "phishing_email", 2: "legitimate_url", 3: "phishing_url"}

_FILES = ["data/train-00000-of-00002.parquet"]  # 120k rows is plenty; idempotent


def fetch_files(session) -> list[pathlib.Path]:
    from scamless.data.download import fetch

    paths = []
    for rel in _FILES:
        url = f"https://huggingface.co/datasets/{HF_ID}/resolve/main/{rel}"
        dest = RAW / "phishing_v2" / pathlib.Path(rel).name
        paths.append(fetch(url, dest, session))
    return paths


def records_from_dataframe(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    messages: list[dict] = []
    url_records: list[dict] = []
    content = df["content"].astype("string").fillna("")
    labels = df["label"].astype("Int64").fillna(0)
    for text, label in zip(content, labels):
        text = str(text).strip()
        kind = _LABEL_MAP.get(int(label))
        if kind is None or len(text) < MIN_TEXT_CHARS:
            continue
        if kind == "phishing_url":
            url_records.append(make_url(text, malicious=True, source=SOURCE))
        elif kind == "legitimate_url":
            url_records.append(make_url(text, malicious=False, source=SOURCE))
        elif kind == "phishing_email":
            messages.append(
                make_message(text, ["phishing"], f"{SOURCE}_email", language="en")
            )
        else:
            messages.append(make_message(text, [], f"{SOURCE}_email", language="en"))
    return messages, url_records


def collect() -> tuple[list[dict], list[dict]]:
    all_messages: list[dict] = []
    all_urls: list[dict] = []
    v2_dir = RAW / "phishing_v2"
    for shard in sorted(v2_dir.glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["content", "label"])
        msgs, urls = records_from_dataframe(df)
        all_messages.extend(msgs)
        all_urls.extend(urls)
    return all_messages, all_urls
