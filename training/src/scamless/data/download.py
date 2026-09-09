"""Thin download helpers. Network code lives here and only here."""

import pathlib
import tarfile
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


def fetch(url: str, dest: pathlib.Path, session: requests.Session | None = None) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    s = session or requests.Session()
    resp = s.get(url, timeout=180)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def extract_archive(archive: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    return dest_dir
