"""Download all raw corpora into training/data/raw/. Run once; idempotent.

Fault isolation: one corpus failing does not abort the others; a summary
reports what failed so the build step can degrade gracefully and a re-run
retries only the missing pieces.

Usage: python -m scamless.data.fetch_all
"""

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

    failures: list[str] = []

    def guarded(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            failures.append(f"{name}: {exc}")
            print(f"FETCH FAILED ({name}): {exc}", flush=True)

    def _sms() -> None:
        sms_zip = fetch(SMS_ZIP, RAW / "sms" / "sms.zip", session)
        extract_archive(sms_zip, RAW / "sms")

    def _spamassassin() -> None:
        for name, url in SPAMASSASSIN.items():
            tarball = fetch(url, RAW / "spamassassin" / f"{name}.tar.bz2", session)
            extract_archive(tarball, RAW / "spamassassin")

    guarded("sms", _sms)
    guarded("spamassassin", _spamassassin)
    guarded("openphish", lambda: fetch(OPENPHISH, RAW / "urls" / "openphish.txt", session))
    guarded(
        "majestic",
        lambda: fetch(MAJESTIC, RAW / "urls" / "majestic_million.csv", session),
    )
    guarded(
        "hf_phishing_texts",
        lambda: fetch(HF_PHISHING_TEXTS, RAW / "phishing_hf" / "texts.json", session),
    )

    print(f"Raw corpora ready under {RAW}")
    if failures:
        print(f"{len(failures)} corpus download(s) failed: {failures}")
        print("The build step will skip the missing sources; re-run fetch_all to retry.")


if __name__ == "__main__":
    run()
