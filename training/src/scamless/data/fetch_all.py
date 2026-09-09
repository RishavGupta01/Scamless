"""Download all raw corpora into training/data/raw/. Run once; idempotent.

Usage: python -m scamless.data.fetch_all
"""

import pathlib

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


def run() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "scamless-training/0.1"

    sms_zip = fetch(SMS_ZIP, RAW / "sms" / "sms.zip", session)
    extract_archive(sms_zip, RAW / "sms")

    for name, url in SPAMASSASSIN.items():
        tarball = fetch(url, RAW / "spamassassin" / f"{name}.tar.bz2", session)
        extract_archive(tarball, RAW / "spamassassin")

    fetch(OPENPHISH, RAW / "urls" / "openphish.txt", session)
    fetch(MAJESTIC, RAW / "urls" / "majestic_million.csv", session)
    fetch(HF_PHISHING_TEXTS, RAW / "phishing_hf" / "texts.json", session)

    print(f"Raw corpora ready under {RAW}")


if __name__ == "__main__":
    run()
