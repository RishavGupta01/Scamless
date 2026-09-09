"""Thin download helpers. Network code lives here and only here.

Safety properties:
- Atomic writes: files land as .part then rename, so a crashed download can
  never be mistaken for a complete one on re-run.
- Retries with backoff for flaky hosts.
"""


import pathlib
import tarfile
import time
import zipfile

import requests

RAW = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"

SMS_ZIP = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAMASSASSIN = {
    "spam_2": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    "easy_ham_2": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "hard_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
}
OPENPHISH = "https://openphish.com/feed.txt"
MAJESTIC = "https://downloads.majestic.com/majestic_million.csv"
HF_PHISHING_TEXTS = (
    "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/texts.json"
)


def fetch(
    url: str,
    dest: pathlib.Path,
    session: requests.Session | None = None,
    attempts: int = 3,
    backoff_seconds: float = 5.0,
) -> pathlib.Path:
    """Download with retry + atomic write. Returns dest, raises on final failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    s = session or requests.Session()
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = s.get(url, timeout=180)
            resp.raise_for_status()
            part.write_bytes(resp.content)
            part.replace(dest)  # atomic: complete file appears only once valid
            return dest
        except Exception as exc:  # noqa: BLE001 - retry any transport/parse error
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < attempts:
                print(f"retry {attempt}/{attempts - 1} for {url}: {exc}", flush=True)
                time.sleep(backoff_seconds)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error


def extract_archive(archive: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"unrecognized archive format: {archive}")
    return dest_dir
