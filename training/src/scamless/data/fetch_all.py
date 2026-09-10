"""Download all raw corpora into training/data/raw/. Run once; idempotent.

Fault isolation: one corpus failing does not abort the others. Downloads run
in parallel threads (each with its own session, so thread safety is by
construction) and a summary reports what failed so the build step can
degrade gracefully; a re-run retries only the missing pieces.

Usage: python -m scamless.data.fetch_all
"""

import pathlib
from concurrent.futures import ThreadPoolExecutor

import requests

from scamless.data.download import (
    HF_PHISHING_TEXTS,
    MAJESTIC,
    OPENPHISH,
    RAW,
    SMS_ZIP,
    SPAMASSASSIN,
    extract_archive,
    fetch,
)

MULTILINGUAL_SMS = "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset"
PHISHING_V2 = "cybersectony/PhishingEmailDetectionv2.0"
PHISHING_V2_FILE = "data/train-00000-of-00001.parquet"
SEVEN_PHISHING = "puyang2025/seven-phishing-email-datasets"
SEVEN_PHISHING_FILE = "train.parquet"


def run() -> None:
    failures: list[str] = []

    def job(name: str, fn) -> None:
        try:
            session = requests.Session()
            session.headers["User-Agent"] = "scamless-training/0.1"
            fn(session)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            failures.append(f"{name}: {exc}")
            print(f"FETCH FAILED ({name}): {exc}", flush=True)

    def _sms(session) -> None:
        sms_zip = fetch(SMS_ZIP, RAW / "sms" / "sms.zip", session)
        extract_archive(sms_zip, RAW / "sms")

    def _spamassassin(session) -> None:
        for name, url in SPAMASSASSIN.items():
            tarball = fetch(url, RAW / "spamassassin" / f"{name}.tar.bz2", session)
            extract_archive(tarball, RAW / "spamassassin")

    def _seven_phishing(session) -> None:
        url = f"https://huggingface.co/datasets/{SEVEN_PHISHING}/resolve/main/{SEVEN_PHISHING_FILE}"
        fetch(url, RAW / "seven_phishing" / SEVEN_PHISHING_FILE, session)

    jobs = [
        ("sms", _sms),
        ("spamassassin", _spamassassin),
        ("openphish", lambda s: fetch(OPENPHISH, RAW / "urls" / "openphish.txt", s)),
        ("majestic", lambda s: fetch(MAJESTIC, RAW / "urls" / "majestic_million.csv", s)),
        (
            "hf_phishing_texts",
            lambda s: fetch(HF_PHISHING_TEXTS, RAW / "phishing_hf" / "texts.json", s),
        ),
        (
            "multilingual_sms",
            lambda s: fetch(
                f"https://huggingface.co/datasets/{MULTILINGUAL_SMS}/resolve/main/data-augmented.csv",
                RAW / "multilingual_sms" / "data-augmented.csv",
                s,
            ),
        ),
        (
            "phishing_v2",
            lambda s: fetch(
                f"https://huggingface.co/datasets/{PHISHING_V2}/resolve/main/{PHISHING_V2_FILE}",
                RAW / "phishing_v2" / pathlib.Path(PHISHING_V2_FILE).name,
                s,
            ),
        ),
        ("seven_phishing", _seven_phishing),
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda j: job(j[0], j[1]), jobs))

    print(f"Raw corpora ready under {RAW}")
    if failures:
        print(f"{len(failures)} corpus download(s) failed: {failures}")
        print("The build step will skip the missing sources; re-run fetch_all to retry.")


if __name__ == "__main__":
    run()
