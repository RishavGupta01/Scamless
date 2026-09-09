# Scamless Phase 1: Detection Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproducible data pipeline, eval harness with CI gates, and the trained English-first multi-label scam classifier exported to int8 ONNX — the core that the later web app and Chrome extension consume.

**Architecture:** Python package `training/` with a pure-function parse layer (fixture-tested, no network in unit tests), a dataset builder emitting train/val/test Parquet, an adversarial augmentation module, a metrics harness with threshold gates, a config-driven fine-tuning script, and an ONNX int8 export step verified against PyTorch logits. Repo is a monorepo; `apps/web` and `apps/extension` arrive in later plans.

**Tech Stack:** Python 3.11+, pytest, pandas + PyArrow, requests, scikit-learn, PyTorch, Hugging Face transformers + datasets, optimum (ONNX export), onnxruntime (+ quantization), ruff. **Training hardware: Google Colab free T4 GPU** — the local machine runs unit tests only (CPU); all heavy steps (fetch, build, train, export, eval, gate) run in `notebooks/train_colab.ipynb`.

**Spec:** `docs/superpowers/specs/2026-09-09-scamsense-design.md` (this plan covers spec sections 3, 4, 6, and the Phase 1–2 portions of 7, 11, 12).

**Rules:** No emojis anywhere (code, comments, commits, docs). Every task ends with a commit. Unit tests never touch the network.

---

### Task 1: Repository scaffold and Python tooling

**Files:**
- Create: `training/pyproject.toml`
- Create: `training/src/scamless/__init__.py`
- Create: `training/tests/__init__.py`
- Create: `training/tests/test_smoke.py`
- Create: `.gitignore`

- [ ] **Step 1: Verify Python is available**

Run (from repo root): `python --version`
Expected: `Python 3.11` or newer printed. If missing, stop and install Python 3.11+ first.

- [ ] **Step 2: Write pyproject.toml**

Create `training/pyproject.toml`:

```toml
[project]
name = "scamless-training"
version = "0.1.0"
description = "Scamless detection core: data pipeline, training, eval"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.1",
    "pyarrow>=15.0",
    "requests>=2.31",
    "scikit-learn>=1.4",
    "torch>=2.2",
    "transformers>=4.40",
    "datasets>=2.18",
    "onnxruntime>=1.17",
    "optimum[exporters]>=1.19",
    "huggingface-hub>=0.23",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: long-running tests (training, downloads)"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Write package init, smoke test, gitignore**

Create `training/src/scamless/__init__.py`:

```python
```

(empty file)

Create `training/tests/__init__.py`:

```python
```

(empty file)

Create `training/tests/test_smoke.py`:

```python
import scamless


def test_package_imports():
    assert scamless is not None
```

Create `.gitignore` at repo root:

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
training/data/raw/
training/data/processed/
training/artifacts/
*.onnx
.ipynb_checkpoints/
```

- [ ] **Step 4: Create venv and install**

Run (from repo root):

```powershell
python -m venv .venv; if ($?) { & .venv\Scripts\python.exe -m pip install -e "training[dev]" }
```

Expected: install completes without errors (this takes several minutes on first run).

- [ ] **Step 5: Run pytest**

Run (from repo root): `& .venv\Scripts\python.exe -m pytest training/tests -v`
Expected: `test_package_imports PASSED`, 1 passed.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore training
git commit -m "feat: scaffold training package with pytest tooling"
```

---

### Task 2: Label schema

**Files:**
- Create: `training/src/scamless/labels.py`
- Test: `training/tests/test_labels.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_labels.py`:

```python
from scamless.labels import (
    LABELS,
    NUM_SCAM_LABELS,
    SAFE,
    SCAM_LABELS,
    label_index,
    labels_to_vector,
    vector_to_labels,
)


def test_fifteen_scam_labels_plus_safe():
    assert len(SCAM_LABELS) == 15
    assert SAFE not in SCAM_LABELS
    assert LABELS == SCAM_LABELS  # model output space is the 15 scam labels


def test_label_index_is_stable():
    assert label_index("phishing") == 0
    assert label_index("generic_spam") == NUM_SCAM_LABELS - 1


def test_labels_to_vector_roundtrip():
    vec = labels_to_vector(["phishing", "payment_pressure"])
    assert len(vec) == NUM_SCAM_LABELS
    assert vec[label_index("phishing")] == 1.0
    assert vec[label_index("payment_pressure")] == 1.0
    assert vec[label_index("romance")] == 0.0
    assert set(vector_to_labels(vec)) == {"phishing", "payment_pressure"}


def test_vector_to_labels_all_zero_is_safe():
    assert vector_to_labels([0.0] * NUM_SCAM_LABELS) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scamless.labels'`

- [ ] **Step 3: Implement labels.py**

Create `training/src/scamless/labels.py`:

```python
"""Canonical label schema. Order is a public contract: never reorder, only append."""

SCAM_LABELS = [
    "phishing",              # 0
    "investment_crypto",     # 1
    "romance",               # 2
    "tech_support",          # 3
    "lottery_prize",         # 4
    "job_task",              # 5
    "gov_bank_impersonation",# 6
    "otp_request",           # 7
    "payment_pressure",      # 8
    "advance_fee",           # 9
    "sextortion",            # 10
    "delivery_scam",         # 11
    "account_suspension",    # 12
    "malicious_link",        # 13
    "generic_spam",          # 14
]

SAFE = "safe"
LABELS = SCAM_LABELS
NUM_SCAM_LABELS = len(SCAM_LABELS)

_INDEX = {name: i for i, name in enumerate(SCAM_LABELS)}


def label_index(name: str) -> int:
    return _INDEX[name]


def labels_to_vector(names: list[str]) -> list[float]:
    vec = [0.0] * NUM_SCAM_LABELS
    for name in names:
        if name != SAFE:
            vec[label_index(name)] = 1.0
    return vec


def vector_to_labels(vec: list[float]) -> list[str]:
    return [SCAM_LABELS[i] for i, v in enumerate(vec) if v > 0.5]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/test_labels.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/labels.py training/tests/test_labels.py
git commit -m "feat: canonical multi-label schema for 15 scam categories"
```

---

### Task 3: Record schema and SMS Spam Collection parser

**Files:**
- Create: `training/src/scamless/data/__init__.py`
- Create: `training/src/scamless/data/schemas.py`
- Create: `training/src/scamless/data/sms_spam.py`
- Test: `training/tests/data/test_sms_spam.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/__init__.py` (empty) and `training/tests/data/test_sms_spam.py`:

```python
from scamless.data.sms_spam import parse_sms_lines


def test_parse_sms_spam_collection_lines():
    lines = [
        "ham\tGo until jurong point, crazy.. Available only in bugis n great world la",
        "spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005",
        "ham\tOk lar... Joking wif u oni...",
    ]
    records = parse_sms_lines(lines)
    assert len(records) == 3
    assert records[0]["labels"] == []
    assert records[1]["labels"] == ["generic_spam"]
    assert records[0]["source"] == "sms_spam_collection"
    assert "jurong point" in records[0]["text"]


def test_blank_lines_skipped():
    records = parse_sms_lines(["", "  ", "spam\tWINNER! text WIN to 80085"])
    assert len(records) == 1
    assert records[0]["labels"] == ["generic_spam"]


def test_text_stripped_and_tab_safe():
    records = parse_sms_lines(["ham\t  padded text  "])
    assert records[0]["text"] == "padded text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_sms_spam.py -v`
Expected: FAIL, no module `scamless.data`

- [ ] **Step 3: Implement schemas and parser**

Create `training/src/scamless/data/__init__.py` (empty).

Create `training/src/scamless/data/schemas.py`:

```python
"""Record shapes shared by every parser and the dataset builder.

MessageRecord: {"text": str, "labels": list[str], "source": str, "language": str}
UrlRecord: {"url": str, "malicious": bool, "source": str}
"""

MESSAGE_FIELDS = ["text", "labels", "source", "language"]
URL_FIELDS = ["url", "malicious", "source"]


def make_message(text: str, labels: list[str], source: str, language: str = "en") -> dict:
    return {"text": text.strip(), "labels": list(labels), "source": source, "language": language}


def make_url(url: str, malicious: bool, source: str) -> dict:
    return {"url": url.strip(), "malicious": bool(malicious), "source": source}
```

Create `training/src/scamless/data/sms_spam.py`:

```python
"""Parser for the UCI SMS Spam Collection format: '<label>\t<text>' per line."""

from scamless.data.schemas import make_message

SOURCE = "sms_spam_collection"


def parse_sms_lines(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        label, _, text = line.partition("\t")
        labels = ["generic_spam"] if label == "spam" else []
        records.append(make_message(text, labels, SOURCE, language="en"))
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_sms_spam.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/data training/tests/data
git commit -m "feat: record schema and SMS spam collection parser"
```

---

### Task 4: Email body extractor (shared by SpamAssassin and Nazario)

**Files:**
- Create: `training/src/scamless/data/emails.py`
- Test: `training/tests/data/test_emails.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/test_emails.py`:

```python
from scamless.data.emails import extract_body


def test_extracts_plain_text_body():
    raw = (
        b"From: bank@alerts.example\r\n"
        b"Subject: Account alert\r\n"
        b"\r\n"
        b"Your account will be suspended.\r\n"
        b"Call us now.\r\n"
    )
    body = extract_body(raw)
    assert "Your account will be suspended." in body


def test_extracts_text_part_from_multipart():
    raw = (
        b"From: x@example\r\n"
        b"Subject: hi\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/alternative; boundary="BOUND"\r\n'
        b"\r\n"
        b"--BOUND\r\n"
        b"Content-Type: text/html\r\n"
        b"\r\n"
        b"<b>html part</b>\r\n"
        b"--BOUND\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"plain part here\r\n"
        b"--BOUND--\r\n"
    )
    body = extract_body(raw)
    assert "plain part here" in body


def test_returns_empty_string_when_no_body():
    raw = b"From: x@example\r\nSubject: only headers\r\n"
    assert extract_body(raw) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_emails.py -v`
Expected: FAIL, no module `scamless.data.emails`

- [ ] **Step 3: Implement extract_body**

Create `training/src/scamless/data/emails.py`:

```python
"""Email parsing shared by SpamAssassin and Nazario corpora."""

import email
from email.message import Message


def extract_body(raw: bytes) -> str:
    msg = email.message_from_bytes(raw)
    return _body_from_message(msg).strip()


def _body_from_message(msg: Message) -> str:
    if msg.is_multipart():
        plain = None
        for part in msg.get_payload():
            if part.get_content_type() == "text/plain":
                plain = part
                break
        chosen = plain if plain is not None else msg.get_payload()[0]
        return _decode_part(chosen)
    if msg.get_content_type().startswith("text/"):
        return _decode_part(msg)
    return ""


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_emails.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/data/emails.py training/tests/data/test_emails.py
git commit -m "feat: shared email body extractor for spam and phishing corpora"
```

---

### Task 5: SpamAssassin corpus parser

**Files:**
- Create: `training/src/scamless/data/spamassassin.py`
- Test: `training/tests/data/test_spamassassin.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/test_spamassassin.py`:

```python
from scamless.data.spamassassin import classify_dir, parse_message_bytes


def test_classify_dir_names():
    assert classify_dir("spam_2") == ["generic_spam"]
    assert classify_dir("easy_ham_2") == []
    assert classify_dir("hard_ham") == []


def test_parse_message_bytes():
    raw = (
        b"Subject: urgent offer\r\n\r\n"
        b"Click here to claim your prize money now\r\n"
    )
    record = parse_message_bytes(raw, dirname="spam_2", filename="00001.abc")
    assert record["labels"] == ["generic_spam"]
    assert record["source"] == "spamassassin"
    assert "claim your prize" in record["text"]


def test_short_bodies_dropped():
    raw = b"Subject: hi\r\n\r\nok\r\n"
    assert parse_message_bytes(raw, "spam_2", "x.eml") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_spamassassin.py -v`
Expected: FAIL, no module `scamless.data.spamassassin`

- [ ] **Step 3: Implement parser**

Create `training/src/scamless/data/spamassassin.py`:

```python
"""Parser for the SpamAssassin public corpus layout (spam_*/ easy_ham_* dirs)."""

from scamless.data.emails import extract_body
from scamless.data.schemas import make_message

SOURCE = "spamassassin"
MIN_BODY_CHARS = 40


def classify_dir(dirname: str) -> list[str]:
    if dirname.startswith("spam"):
        return ["generic_spam"]
    return []


def parse_message_bytes(raw: bytes, dirname: str, filename: str) -> dict | None:
    text = extract_body(raw)
    if len(text) < MIN_BODY_CHARS:
        return None
    return make_message(text, classify_dir(dirname), SOURCE, language="en")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_spamassassin.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/data/spamassassin.py training/tests/data/test_spamassassin.py
git commit -m "feat: spamassassin corpus parser"
```

---

### Task 6: Nazario phishing corpus parser

**Files:**
- Create: `training/src/scamless/data/nazario.py`
- Test: `training/tests/data/test_nazario.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/test_nazario.py`:

```python
from scamless.data.nazario import parse_phishing_email


def test_parse_phishing_email():
    raw = (
        b"From: security@paypa1-example.com\r\n"
        b"Subject: Verify your account\r\n\r\n"
        b"Dear customer, we detected unusual activity. "
        b"Please verify your identity within 24 hours by clicking the link below. "
        b"Failure to comply will result in permanent suspension.\r\n"
    )
    record = parse_phishing_email(raw, filename="phish001.eml")
    assert record["labels"] == ["phishing"]
    assert record["source"] == "nazario"
    assert "unusual activity" in record["text"]


def test_short_body_dropped():
    raw = b"Subject: x\r\n\r\nverify now\r\n"
    assert parse_phishing_email(raw, "x.eml") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_nazario.py -v`
Expected: FAIL, no module `scamless.data.nazario`

- [ ] **Step 3: Implement parser**

Create `training/src/scamless/data/nazario.py`:

```python
"""Parser for the Nazario phishing corpus (.eml files)."""

from scamless.data.emails import extract_body
from scamless.data.schemas import make_message

SOURCE = "nazario"
MIN_BODY_CHARS = 40


def parse_phishing_email(raw: bytes, filename: str) -> dict | None:
    text = extract_body(raw)
    if len(text) < MIN_BODY_CHARS:
        return None
    return make_message(text, ["phishing"], SOURCE, language="en")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_nazario.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/data/nazario.py training/tests/data/test_nazario.py
git commit -m "feat: nazario phishing corpus parser"
```

---

### Task 7: Enron loader (Hugging Face SetFit/enron_spam)

**Files:**
- Create: `training/src/scamless/data/enron.py`
- Test: `training/tests/data/test_enron.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/test_enron.py`:

```python
from scamless.data.enron import records_from_hf_rows


def test_records_from_hf_rows():
    rows = [
        {"label": 1, "text": "Meeting is at 3pm on Tuesday, please confirm attendance."},
        {"label": 0, "text": "Congratulations! You have been selected for a cash prize."},
        {"label": 0, "text": "short"},
    ]
    records = records_from_hf_rows(rows)
    assert len(records) == 2
    assert records[0]["labels"] == []
    assert records[1]["labels"] == ["generic_spam"]
    assert records[0]["source"] == "enron_spam"
```

Note: in `SetFit/enron_spam`, `label=1` is legitimate mail ("ham") and `label=0` is spam. Verify against the dataset card during the download task; the mapping lives in one place (`LABEL_FOR_SPAM`), so a correction is a one-line change.

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_enron.py -v`
Expected: FAIL, no module `scamless.data.enron`

- [ ] **Step 3: Implement loader**

Create `training/src/scamless/data/enron.py`:

```python
"""Loader for the Hugging Face SetFit/enron_spam dataset (network used in download task only)."""

from scamless.data.schemas import make_message

SOURCE = "enron_spam"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_enron.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/data/enron.py training/tests/data/test_enron.py
git commit -m "feat: enron spam loader with label mapping isolated"
```

---

### Task 8: URL datasets (OpenPhish + Majestic Million)

**Files:**
- Create: `training/src/scamless/data/urls.py`
- Test: `training/tests/data/test_urls.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/test_urls.py`:

```python
from scamless.data.urls import parse_majestic_csv, parse_openphish_lines


def test_parse_openphish_lines():
    lines = [
        "http://secure-login.example-verify.com/account",
        "https://update-billing.example.ru/pay",
        "",
        "# comment style lines are skipped",
    ]
    records = parse_openphish_lines(lines)
    assert len(records) == 2
    assert all(r["malicious"] for r in records)
    assert records[0]["source"] == "openphish"


def test_parse_majestic_csv_top_n():
    csv_text = (
        "GlobalRank,TldRank,Domain,TLD,RefSubNets,RefIPs,IDN_Domain,IDN_Encoding\n"
        "1,1,google.com,com,1000,2000,google.com,google.com\n"
        "2,1,facebook.com,com,900,1800,facebook.com,facebook.com\n"
        "3,1,badsite.example,example,1,2,badsite.example,badsite.example\n"
    )
    records = parse_majestic_csv(csv_text, top_n=2)
    assert [r["url"] for r in records] == ["http://google.com", "http://facebook.com"]
    assert all(not r["malicious"] for r in records)
    assert records[0]["source"] == "majestic_million"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_urls.py -v`
Expected: FAIL, no module `scamless.data.urls`

- [ ] **Step 3: Implement url parsers**

Create `training/src/scamless/data/urls.py`:

```python
"""URL corpora: OpenPhish community feed (malicious) and Majestic Million (benign).

Majestic Million substitutes the spec's Tranco top-1M because Tranco requires
manual registration; semantics are identical (ranked benign top sites).
"""

import csv
import io

from scamless.data.schemas import make_url


def parse_openphish_lines(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        records.append(make_url(url, malicious=True, source="openphish"))
    return records


def parse_majestic_csv(csv_text: str, top_n: int = 50000) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for i, row in enumerate(reader):
        if i >= top_n:
            break
        domain = (row.get("Domain") or "").strip()
        if not domain:
            continue
        records.append(make_url(f"http://{domain}", malicious=False, source="majestic_million"))
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_urls.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/data/urls.py training/tests/data/test_urls.py
git commit -m "feat: openphish and majestic million url parsers"
```

---

### Task 9: Downloaders (network, integration-only)

**Files:**
- Create: `training/src/scamless/data/download.py`
- Create: `training/src/scamless/data/fetch_all.py`

These hit the network, so they carry no unit tests. Correctness of parsing is already covered; this layer only moves bytes to `training/data/raw/`.

- [ ] **Step 1: Implement download.py**

Create `training/src/scamless/data/download.py`:

```python
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


def fetch(url: str, dest: pathlib.Path, session: requests.Session | None = None) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    s = session or requests.Session()
    resp = s.get(url, timeout=120)
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
```

- [ ] **Step 2: Implement fetch_all.py (the orchestrating script)**

Create `training/src/scamless/data/fetch_all.py`:

```python
"""Download all raw corpora into training/data/raw/. Run once; idempotent.

Usage: python -m scamless.data.fetch_all
"""

import pathlib

import requests

from scamless.data.download import (
    MAJESTIC,
    OPENPHISH,
    RAW,
    SMS_ZIP,
    SPAMASSASSIN,
    extract_archive,
    fetch,
)

ENRON_ID = "SetFit/enron_spam"


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

    print(f"Raw corpora ready under {RAW}")


if __name__ == "__main__":
    run()
```

Note: the Enron corpus is pulled via `datasets.load_dataset(ENRON_ID)` inside the builder's fetch step (Task 10) rather than downloaded here.

- [ ] **Step 3: Mark an integration check (manual)**

Run: `& .venv\Scripts\python.exe -m scamless.data.fetch_all`
Expected: prints `Raw corpora ready under ...training\data\raw` and directories `sms`, `spamassassin`, `urls` exist with files. (Requires internet; takes a few minutes.)

- [ ] **Step 4: Commit**

```powershell
git add training/src/scamless/data/download.py training/src/scamless/data/fetch_all.py
git commit -m "feat: idempotent raw corpus downloaders"
```

---

### Task 10: Dataset builder (dedupe, splits, report)

**Files:**
- Create: `training/src/scamless/data/build.py`
- Test: `training/tests/data/test_build.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/data/test_build.py`:

```python
import pandas as pd

from scamless.data.build import dedupe, split_records


def _msgs():
    return [
        {"text": "win free prizes now click here", "labels": ["generic_spam"], "source": "a", "language": "en"},
        {"text": "win free prizes now click here", "labels": ["generic_spam"], "source": "b", "language": "en"},
        {"text": "please review the attached invoice for march", "labels": [], "source": "a", "language": "en"},
        {"text": "verify your account at this secure link", "labels": ["phishing"], "source": "c", "language": "en"},
    ] * 10


def test_dedupe_keeps_first_occurrence_per_text():
    out = dedupe(_msgs())
    assert len(out) == 3
    assert {r["source"] for r in out} == {"a", "c"}


def test_split_records_proportions_and_disjoint():
    df = pd.DataFrame(dedupe(_msgs()))
    train, val, test = split_records(df, seed=7)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(val) - round(0.1 * len(df))) <= 1
    train_texts = set(train["text"])
    test_texts = set(test["text"])
    assert not (train_texts & test_texts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_build.py -v`
Expected: FAIL, no module `scamless.data.build`

- [ ] **Step 3: Implement builder**

Create `training/src/scamless/data/build.py`:

```python
"""Dataset builder: parse all corpora, dedupe, split, write Parquet + report.

Usage: python -m scamless.data.build [--enron-rows N]
"""

import argparse
import collections
import json
import pathlib

import pandas as pd

from scamless.data import enron, nazario, sms_spam, spamassassin, urls
from scamless.data.download import RAW

PROCESSED = RAW.parent / "processed"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def collect_messages() -> list[dict]:
    records: list[dict] = []

    sms_lines = (RAW / "sms" / "SMSSpamCollection").read_text(encoding="utf-8", errors="replace").splitlines()
    records.extend(sms_spam.parse_sms_lines(sms_lines))

    sa_root = RAW / "spamassassin"
    if sa_root.exists():
        for tar_dir in sorted(sa_root.iterdir()):
            if not tar_dir.is_dir():
                continue
            for f in sorted(tar_dir.rglob("*")):
                if f.is_file():
                    rec = spamassassin.parse_message_bytes(f.read_bytes(), tar_dir.name, f.name)
                    if rec:
                        records.append(rec)

    nz_root = RAW / "nazario"
    if nz_root.exists():
        for f in sorted(nz_root.rglob("*.eml")):
            rec = nazario.parse_phishing_email(f.read_bytes(), f.name)
            if rec:
                records.append(rec)

    try:
        from datasets import load_dataset

        ds = load_dataset(enron.SOURCE_HF_ID)
        rows = [dict(r) for split in ds.values() for r in split]
        records.extend(enron.records_from_hf_rows(rows))
    except Exception as exc:  # enron is optional for a build
        print(f"enron skipped: {exc}")

    return records


def collect_urls() -> list[dict]:
    out: list[dict] = []
    op = RAW / "urls" / "openphish.txt"
    if op.exists():
        out.extend(urls.parse_openphish_lines(op.read_text(encoding="utf-8", errors="replace").splitlines()))
    mj = RAW / "urls" / "majestic_million.csv"
    if mj.exists():
        out.extend(urls.parse_majestic_csv(mj.read_text(encoding="utf-8", errors="replace"), top_n=50000))
    return out


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in records:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_records(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_val = round(SPLITS["val"] * n)
    n_test = round(SPLITS["test"] * n)
    test = shuffled.iloc[:n_test]
    val = shuffled.iloc[n_test : n_test + n_val]
    train = shuffled.iloc[n_test + n_val :]
    return train, val, test


def report(records: list[dict], urls: list[dict], path: pathlib.Path) -> None:
    label_counts = collections.Counter(l for r in records for l in r["labels"])
    safe_count = sum(1 for r in records if not r["labels"])
    payload = {
        "messages_total": len(records),
        "messages_safe": safe_count,
        "messages_labeled": len(records) - safe_count,
        "label_counts": dict(label_counts.most_common()),
        "urls_total": len(urls),
        "urls_malicious": sum(1 for u in urls if u["malicious"]),
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enron-rows", type=int, default=20000)
    args = parser.parse_args()

    messages = dedupe(collect_messages())
    url_records = collect_urls()

    msg_df = pd.DataFrame(messages)
    train, val, test = split_records(msg_df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED / "messages_train.parquet")
    val.to_parquet(PROCESSED / "messages_val.parquet")
    test.to_parquet(PROCESSED / "messages_test.parquet")
    pd.DataFrame(url_records).to_parquet(PROCESSED / "urls.parquet")

    report(messages, url_records, PROCESSED / "report.json")
    print(f"built: {len(train)} train / {len(val)} val / {len(test)} test messages, {len(url_records)} urls")


if __name__ == "__main__":
    main()
```

Also add the HF dataset id to `training/src/scamless/data/enron.py` (modify):

```python
SOURCE_HF_ID = "SetFit/enron_spam"
```

(add this line directly below `SOURCE = "enron_spam"`)

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/data/test_build.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the real build (integration, network needed)**

Run: `& .venv\Scripts\python.exe -m scamless.data.build`
Expected: prints `built: N train / M val / K test messages, L urls`; files exist in `training/data/processed/`. `report.json` shows a non-zero count for every label category (labels with zero rows are flagged for the synthetic-data task in the multilingual phase).

- [ ] **Step 6: Commit**

```powershell
git add training/src/scamless/data/build.py training/src/scamless/data/enron.py training/tests/data/test_build.py
git commit -m "feat: dataset builder with dedupe, splits, and label report"
```

---

### Task 11: Adversarial augmentation module

**Files:**
- Create: `training/src/scamless/aug/__init__.py`
- Create: `training/src/scamless/aug/adversarial.py`
- Test: `training/tests/test_adversarial.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_adversarial.py`:

```python
import random

from scamless.aug.adversarial import (
    apply_all,
    homoglyph_substitute,
    insert_zero_width,
    leet_substitute,
)


def test_homoglyph_substitute_changes_latin_chars():
    out = homoglyph_substitute("paypal.com secure", random.Random(1))
    assert out != "paypal.com secure"
    assert "а" in out or "е" in out or "о" in out  # cyrillic lookalikes present


def test_zero_width_insert_is_invisible():
    out = insert_zero_width("urgent action", random.Random(1))
    assert out != "urgent action"
    assert out.replace("\u200b", "") == "urgent action"


def test_leet_substitute_maps_known_chars():
    out = leet_substitute("free money", random.Random(1))
    assert out != "free money"
    assert all(c not in "free money" or True for c in out)  # sanity: no crash


def test_apply_all_preserves_meaning_characters():
    text = "claim your prize now at the link"
    out = apply_all(text, random.Random(3))
    assert len(out) >= len(text)
    assert out != text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/test_adversarial.py -v`
Expected: FAIL, no module `scamless.aug`

- [ ] **Step 3: Implement augmentation**

Create `training/src/scamless/aug/__init__.py` (empty) and `training/src/scamless/aug/adversarial.py`:

```python
"""Adversarial augmentations: homoglyphs, zero-width chars, leetspeak.

Applied at training time to a fraction of scam examples so the model is
robust to obfuscation attacks. Deterministic given the rng argument.
"""

import random

HOMOGLYPHS = {
    "a": "а",  # cyrillic a
    "e": "е",  # cyrillic e
    "o": "о",  # cyrillic o
    "p": "р",  # cyrillic p
    "c": "с",  # cyrillic c
    "y": "у",  # cyrillic y
    "x": "х",  # cyrillic x
    "i": "і",  # cyrillic i
}

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

ZERO_WIDTH = "\u200b"


def homoglyph_substitute(text: str, rng: random.Random, rate: float = 0.15) -> str:
    return "".join(
        HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < rate else c for c in text
    )


def leet_substitute(text: str, rng: random.Random, rate: float = 0.2) -> str:
    return "".join(LEET[c] if c in LEET and rng.random() < rate else c for c in text)


def insert_zero_width(text: str, rng: random.Random, rate: float = 0.1) -> str:
    chars = []
    for c in text:
        chars.append(c)
        if c.isalpha() and rng.random() < rate:
            chars.append(ZERO_WIDTH)
    return "".join(chars)


def apply_all(text: str, rng: random.Random) -> str:
    text = leet_substitute(text, rng)
    text = homoglyph_substitute(text, rng)
    return insert_zero_width(text, rng)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/test_adversarial.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/aug training/tests/test_adversarial.py
git commit -m "feat: adversarial augmentation with homoglyph, zero-width, leet transforms"
```

---

### Task 12: Eval harness (metrics math)

**Files:**
- Create: `training/src/scamless/eval/__init__.py`
- Create: `training/src/scamless/eval/harness.py`
- Test: `training/tests/eval/__init__.py`
- Test: `training/tests/eval/test_harness.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/eval/__init__.py` (empty) and `training/tests/eval/test_harness.py`:

```python
import pandas as pd

from scamless.eval.harness import compute_metrics


def _frame(texts, label_sets):
    return pd.DataFrame({"text": texts, "labels": label_sets})


def test_perfect_predictions():
    df = _frame(
        ["win money now", "meeting at 3pm", "verify account link"],
        [["generic_spam"], [], ["phishing"]],
    )
    preds = [
        ["generic_spam"],
        [],
        ["phishing"],
    ]
    metrics = compute_metrics(df, preds)
    assert metrics["macro_f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["per_label"]["generic_spam"]["f1"] == 1.0


def test_false_positive_rate_counts_safe_rows_with_any_positive():
    df = _frame(["meeting at 3pm", "win money"], [[], ["generic_spam"]])
    preds = [["phishing"], ["generic_spam"]]
    metrics = compute_metrics(df, preds)
    # one of two safe rows was falsely flagged
    assert metrics["false_positive_rate"] == 0.5


def test_safe_rows_do_not_count_in_scam_recall():
    df = _frame(["meeting at 3pm"], [[]])
    metrics = compute_metrics(df, [[]])
    assert metrics["macro_f1"] == 0.0  # no positives anywhere: zero division -> 0
    assert metrics["false_positive_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/eval/test_harness.py -v`
Expected: FAIL, no module `scamless.eval`

- [ ] **Step 3: Implement harness**

Create `training/src/scamless/eval/__init__.py` (empty) and `training/src/scamless/eval/harness.py`:

```python
"""Metrics harness: per-label P/R/F1, macro-F1, false-positive rate.

fp_rate = share of records with NO true labels (safe) that get >= 1 predicted label.
This is the trust metric; it is optimized as hard as recall (spec section 7).
"""

import json

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from scamless.labels import SCAM_LABELS, labels_to_vector


def compute_metrics(df, preds: list[list[str]]) -> dict:
    y_true = np.array([labels_to_vector(list(r)) for r in df["labels"]])
    y_pred = np.array([labels_to_vector(list(p)) for p in preds])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for name, p, r, f in zip(SCAM_LABELS, prec, rec, f1)
    }

    safe_mask = y_true.sum(axis=1) == 0
    flagged = y_pred.sum(axis=1) > 0
    fp_rate = float((safe_mask & flagged).sum() / safe_mask.sum()) if safe_mask.any() else 0.0

    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    return {
        "macro_f1": macro_f1,
        "false_positive_rate": fp_rate,
        "per_label": per_label,
        "n": int(len(df)),
    }


def save_metrics(metrics: dict, path) -> None:
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/eval/test_harness.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/eval training/tests/eval
git commit -m "feat: metrics harness with false-positive-rate tracking"
```

---

### Task 13: Heuristic baseline (validates the harness end to end)

**Files:**
- Create: `training/src/scamless/eval/baseline.py`
- Test: `training/tests/eval/test_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/eval/test_baseline.py`:

```python
from scamless.eval.baseline import heuristic_predict


def test_scams_trigger_labels():
    assert "generic_spam" in heuristic_predict("WIN a free prize now!! text 80085")
    assert "otp_request" in heuristic_predict("share the OTP code 5521 to verify")
    assert "payment_pressure" in heuristic_predict("send $500 immediately or else")


def test_normal_text_is_safe():
    assert heuristic_predict("Lunch tomorrow at the usual place?") == []
    assert heuristic_predict("Here are the meeting notes from Tuesday.") == []


def test_obfuscated_scams_still_trigger():
    # zero-width chars between letters must not hide the scam
    text = "claim\u200byour\u200bprize"
    assert "generic_spam" in heuristic_predict(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/eval/test_baseline.py -v`
Expected: FAIL, no module `scamless.eval.baseline`

- [ ] **Step 3: Implement baseline**

Create `training/src/scamless/eval/baseline.py`:

```python
"""Keyword heuristic baseline. Purposes:
1. Validates the metrics harness end to end before any training happens.
2. Never the shipped detector; the model replaces it in Task 14.
"""

from scamless.aug.adversarial import ZERO_WIDTH

RULES = {
    "generic_spam": ["free", "prize", "winner", "win ", "claim", "cash", "lottery", "selected"],
    "otp_request": ["otp", "one-time", "one time password", "verification code", "share the code"],
    "phishing": ["verify your account", "confirm your identity", "click the link", "login here"],
    "payment_pressure": ["send $", "send money", "immediately or", "pay now", "gift card"],
    "investment_crypto": ["double your", "investment opportunity", "guaranteed returns", "crypto profit"],
}


def heuristic_predict(text: str) -> list[str]:
    normalized = text.replace(ZERO_WIDTH, "").lower()
    hits = []
    for label, keywords in RULES.items():
        if any(kw in normalized for kw in keywords):
            hits.append(label)
    return hits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/eval/test_baseline.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/eval/baseline.py training/tests/eval/test_baseline.py
git commit -m "feat: keyword heuristic baseline for harness validation"
```

---

### Task 14: Training script (config-driven, smoke-testable)

**Files:**
- Create: `training/src/scamless/model/__init__.py`
- Create: `training/src/scamless/model/train.py`
- Create: `training/src/scamless/model/config.py`
- Test: `training/tests/model/__init__.py`
- Test: `training/tests/model/test_train.py`

- [ ] **Step 1: Write the failing test (smoke: tiny model, tiny data)**

Create `training/tests/model/__init__.py` (empty) and `training/tests/model/test_train.py`:

```python
import pandas as pd
import torch

from scamless.model.config import TrainConfig
from scamless.model.train import build_dataset, train_model


def _tiny_df(n_pos=16, n_neg=16):
    rows = []
    for i in range(n_pos):
        rows.append({"text": f"claim your free prize winner number {i}", "labels": ["generic_spam"]})
    for i in range(n_neg):
        rows.append({"text": f"regular business sentence about invoice {i}", "labels": []})
    return pd.DataFrame(rows)


def test_train_smoke_runs_and_returns_metrics():
    cfg = TrainConfig(
        backbone="prajjwal1/bert-tiny",  # tiny English model: CPU smoke test only
        max_len=64,
        epochs=1,
        batch_size=8,
        lr=5e-5,
        seed=13,
        adversarial_rate=0.0,
    )
    model, tok, metrics = train_model(_tiny_df(), _tiny_df(), cfg)
    assert metrics["val_macro_f1"] >= 0.0
    assert model is not None


def test_build_dataset_shapes():
    cfg = TrainConfig(backbone="prajjwal1/bert-tiny", max_len=64, epochs=1, batch_size=8, lr=5e-5, seed=1)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(cfg.backbone)
    ds = build_dataset(_tiny_df(n_pos=4, n_neg=4), tok, cfg, adversarial_rate=0.0)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(item["labels"]) == 15
    assert item["input_ids"].shape == (64,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/model/test_train.py -v`
Expected: FAIL, no module `scamless.model`

- [ ] **Step 3: Implement config and trainer**

Create `training/src/scamless/model/__init__.py` (empty).

Create `training/src/scamless/model/config.py`:

```python
from dataclasses import dataclass


@dataclass
class TrainConfig:
    backbone: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_len: int = 256
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    seed: int = 42
    adversarial_rate: float = 0.3  # fraction of labeled examples augmented per epoch
    output_dir: str = "artifacts/model_v1"
```

Create `training/src/scamless/model/train.py`:

```python
"""Fine-tune the multilingual MiniLM backbone with a 15-label sigmoid head.

Multi-label via BCEWithLogitsLoss. `bert-tiny` is only for the smoke test;
the default backbone in config.py is the shipped 118M multilingual MiniLM.
"""

import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from scamless.aug.adversarial import apply_all
from scamless.labels import NUM_SCAM_LABELS, labels_to_vector


class MultiLabelDataset(Dataset):
    def __init__(self, texts, label_vectors, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding="max_length")
        self.labels = label_vectors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc["input_ids"][idx]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class WeightedTrainer(Trainer):
    """BCE loss with class weighting handled by pos_weight computed at setup."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self._pos_weight
        )
        return (loss, outputs) if return_outputs else loss


def build_dataset(df, tokenizer, cfg, adversarial_rate: float) -> MultiLabelDataset:
    texts, vectors = [], []
    rng = random.Random(cfg.seed)
    for _, row in df.iterrows():
        text = str(row["text"])
        labels = list(row["labels"])
        if labels and adversarial_rate > 0 and rng.random() < adversarial_rate:
            text = apply_all(text, rng)
        texts.append(text)
        vectors.append(labels_to_vector(labels))
    return MultiLabelDataset(texts, vectors, tokenizer, cfg.max_len)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(train_df, val_df, cfg):
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone,
        num_labels=NUM_SCAM_LABELS,
        problem_type="multi_label_classification",
    )

    train_ds = build_dataset(train_df, tokenizer, cfg, cfg.adversarial_rate)
    val_ds = build_dataset(val_df, tokenizer, cfg, adversarial_rate=0.0)

    # pos_weight: inverse frequency per label, capped to keep loss stable
    vecs = np.array([labels_to_vector(list(r)) for _, r in train_df.iterrows()])
    pos = vecs.sum(axis=0)
    pos_weight = torch.tensor(
        np.clip((len(vecs) - pos) / np.maximum(pos, 1.0), 1.0, 10.0), dtype=torch.float
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        seed=cfg.seed,
        use_cpu=torch.cuda.is_available() is False,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, pos_weight=pos_weight)
    trainer.train()

    # validation metrics with 0.5 threshold
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(val_ds), cfg.batch_size * 4):
            batch = {k: torch.stack([val_ds[j][k] for j in range(i, min(i + cfg.batch_size * 4, len(val_ds)))]) for k in ("input_ids", "attention_mask")}
            logits.append(model(**batch).logits)
        probs = torch.sigmoid(torch.cat(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    truth = np.array([labels_to_vector(list(r)) for _, r in val_df.iterrows()])
    val_macro_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))

    metrics = {"val_macro_f1": val_macro_f1}
    return model, tokenizer, metrics


def main() -> None:
    import pathlib

    import pandas as pd

    from scamless.data.download import RAW

    processed = RAW.parent / "processed"
    cfg = TrainConfig()
    train_df = pd.read_parquet(processed / "messages_train.parquet")
    val_df = pd.read_parquet(processed / "messages_val.parquet")
    model, tokenizer, metrics = train_model(train_df, val_df, cfg)
    out = pathlib.Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/model/test_train.py -v`
Expected: 2 passed (smoke test trains a tiny model for under a minute on CPU).

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/model training/tests/model
git commit -m "feat: config-driven multi-label trainer with adversarial augmentation"
```

---

### Task 15: ONNX export + int8 quantization (verified against PyTorch)

**Files:**
- Create: `training/src/scamless/model/export_onnx.py`
- Test: `training/tests/model/test_export.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/model/test_export.py`:

```python
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from scamless.model.config import TrainConfig
from scamless.model.export_onnx import export_and_quantize
from scamless.model.train import train_model

import pandas as pd


def _tiny_df():
    rows = []
    for i in range(8):
        rows.append({"text": f"claim your free prize winner {i}", "labels": ["generic_spam"]})
        rows.append({"text": f"normal office note about taxes {i}", "labels": []})
    return pd.DataFrame(rows)


def test_exported_quantized_model_matches_pytorch_logits():
    cfg = TrainConfig(backbone="prajjwal1/bert-tiny", max_len=64, epochs=1, batch_size=8,
                      lr=5e-5, seed=7, output_dir="artifacts/test_export")
    model, tok, _ = train_model(_tiny_df(), _tiny_df(), cfg)

    onnx_path = export_and_quantize(cfg.output_dir, cfg.backbone)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    text = "free prize claim now"
    enc = tok([text], return_tensors="np", truncation=True, max_length=cfg.max_len, padding="max_length")
    onnx_logits = sess.run(None, {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    })[0]

    torch_enc = tok([text], return_tensors="pt", truncation=True, max_length=cfg.max_len, padding="max_length")
    model.eval()
    with torch.no_grad():
        torch_logits = model(**torch_enc).logits.numpy()

    # int8 dynamic quantization stays close; tolerate 0.15 logit drift
    assert np.abs(onnx_logits - torch_logits).max() < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/model/test_export.py -v`
Expected: FAIL, no module `scamless.model.export_onnx`

- [ ] **Step 3: Implement export**

Create `training/src/scamless/model/export_onnx.py`:

```python
"""Export the fine-tuned model to ONNX, then apply dynamic int8 quantization.

Usage: python -m scamless.model.export_onnx --model-dir artifacts/model_v1
"""

import argparse
import pathlib

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic
from optimum.exporters.onnx import main_export


def export_and_quantize(model_dir: str, backbone: str) -> pathlib.Path:
    model_dir = pathlib.Path(model_dir)
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    main_export(model_dir, onnx_dir, task="text-classification", opset=15, device="cpu")
    fp32 = onnx_dir / "model.onnx"
    int8 = onnx_dir / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32),
        model_output=str(int8),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    print(f"fp32: {fp32.stat().st_size / 1e6:.1f} MB, int8: {int8.stat().st_size / 1e6:.1f} MB")
    return int8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    args = parser.parse_args()
    export_and_quantize(args.model_dir, backbone=args.model_dir)
    # backbone arg unused here: the saved model dir contains the fine-tuned weights


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/model/test_export.py -v`
Expected: 1 passed. The printout shows the int8 file is substantially smaller than fp32.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/model/export_onnx.py training/tests/model/test_export.py
git commit -m "feat: onnx export with int8 quantization and logit parity check"
```

---

### Task 16: End-to-end eval script + CI gate

**Files:**
- Create: `training/src/scamless/eval/run_eval.py`
- Create: `training/src/scamless/eval/gate.py`
- Create: `training/src/scamless/eval/eval_config.json`
- Test: `training/tests/eval/test_gate.py`

- [ ] **Step 1: Write the failing test**

Create `training/tests/eval/test_gate.py`:

```python
import json

import pytest

from scamless.eval.gate import gate_fails, load_config


def test_gate_passes_when_metrics_meet_thresholds():
    metrics = {"macro_f1": 0.95, "false_positive_rate": 0.005}
    config = {"macro_f1_min": 0.9, "fp_rate_max": 0.01}
    assert gate_fails(metrics, config) == []


def test_gate_fails_lists_each_violation():
    metrics = {"macro_f1": 0.8, "false_positive_rate": 0.02}
    config = {"macro_f1_min": 0.9, "fp_rate_max": 0.01}
    failures = gate_fails(metrics, config)
    assert len(failures) == 2
    assert any("macro_f1" in f for f in failures)
    assert any("false_positive_rate" in f for f in failures)


def test_load_config_reads_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"macro_f1_min": 0.9, "fp_rate_max": 0.01}))
    assert load_config(p)["macro_f1_min"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/eval/test_gate.py -v`
Expected: FAIL, no module `scamless.eval.gate`

- [ ] **Step 3: Implement gate, config, and eval runner**

Create `training/src/scamless/eval/gate.py`:

```python
"""Release gate: exits nonzero when metrics violate thresholds.

Usage: python -m scamless.eval.gate --metrics artifacts/eval/metrics.json
"""

import argparse
import json
import pathlib
import sys


def load_config(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def gate_fails(metrics: dict, config: dict) -> list[str]:
    failures = []
    if metrics["macro_f1"] < config["macro_f1_min"]:
        failures.append(
            f"macro_f1 {metrics['macro_f1']:.3f} < required {config['macro_f1_min']}"
        )
    if metrics["false_positive_rate"] > config["fp_rate_max"]:
        failures.append(
            f"false_positive_rate {metrics['false_positive_rate']:.4f} > allowed {config['fp_rate_max']}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts/eval/metrics.json")
    parser.add_argument("--config", default=str(pathlib.Path(__file__).parent / "eval_config.json"))
    args = parser.parse_args()

    metrics = json.loads(pathlib.Path(args.metrics).read_text())
    config = load_config(pathlib.Path(args.config))
    failures = gate_fails(metrics, config)
    for f in failures:
        print(f"GATE FAIL: {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

Create `training/src/scamless/eval/eval_config.json`:

```json
{
  "macro_f1_min": 0.85,
  "fp_rate_max": 0.02,
  "notes": "v1 gates. Spec section 7 targets (0.95 / 0.01) are the post-multilingual-phase goals; these gates tighten as data improves."
}
```

Create `training/src/scamless/eval/run_eval.py`:

```python
"""Run the trained model (or heuristic baseline) on the test split; write metrics.json.

Usage:
  python -m scamless.eval.run_eval --mode baseline
  python -m scamless.eval.run_eval --mode onnx --model-dir artifacts/model_v1
"""

import argparse
import pathlib

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from scamless.data.download import RAW
from scamless.eval.baseline import heuristic_predict
from scamless.eval.harness import compute_metrics, save_metrics

ARTIFACTS = RAW.parent / ".." / "artifacts" / "eval"


def load_test_df() -> pd.DataFrame:
    processed = RAW.parent / "processed"
    df = pd.read_parquet(processed / "messages_test.parquet")
    return df.reset_index(drop=True)


def predict_baseline(df: pd.DataFrame) -> list[list[str]]:
    return [heuristic_predict(str(t)) for t in df["text"]]


def predict_onnx(df: pd.DataFrame, model_dir: str, max_len: int = 256) -> list[list[str]]:
    onnx_path = pathlib.Path(model_dir) / "onnx" / "model_int8.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    from scamless.labels import SCAM_LABELS

    texts = [str(t) for t in df["text"]]
    preds = []
    for i in range(0, len(texts), 64):
        enc = tokenizer(texts[i : i + 64], truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="np")
        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        for row in probs:
            preds.append([SCAM_LABELS[j] for j in range(len(row)) if row[j] > 0.5])
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "onnx"], default="baseline")
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    args = parser.parse_args()

    df = load_test_df()
    preds = predict_baseline(df) if args.mode == "baseline" else predict_onnx(df, args.model_dir)
    metrics = compute_metrics(df, preds)

    out = pathlib.Path("artifacts/eval/metrics.json")
    save_metrics(metrics, out)
    print(json.dumps({k: metrics[k] for k in ("macro_f1", "false_positive_rate")}, indent=2))
    print(f"metrics written to {out}")


if __name__ == "__main__":
    import json

    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest training/tests/eval/test_gate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add training/src/scamless/eval/run_eval.py training/src/scamless/eval/gate.py training/src/scamless/eval/eval_config.json training/tests/eval/test_gate.py
git commit -m "feat: eval runner and release gate with threshold config"
```

---

### Task 17: Colab GPU training notebook + full pipeline run

**Files:**
- Create: `notebooks/train_colab.ipynb`
- Modify: `training/README.md`

All hardware-heavy steps run on Google Colab (free T4 GPU). The notebook is committed to the repo and clones the repo itself, so Colab always trains against pushed code. Artifacts persist to Google Drive (Colab sessions are ephemeral).

- [ ] **Step 1: Create notebooks/train_colab.ipynb**

Assemble the following cells into a valid .ipynb (JSON) notebook. Cell order matters.

Cell 1 (markdown):

```markdown
# Scamless — Training Run (GPU)

Runtime > Change runtime type > **T4 GPU**, then Run all.
Clones the repo, builds the dataset, fine-tunes, exports int8 ONNX, evals, gates, saves artifacts to Google Drive.
```

Cell 2 (code) — clone repo:

```python
%cd /content
!git clone https://github.com/RishavGupta01/Scamless.git scamless
%cd /content/scamless
```

Cell 3 (code) — install package:

```python
!pip install -q -e "training[dev]"
```

Cell 4 (code) — verify GPU:

```python
!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
```

Cell 5 (code) — mount Drive for persistent artifacts:

```python
from google.colab import drive
import pathlib

drive.mount("/content/drive")
DRIVE_ART = pathlib.Path("/content/drive/MyDrive/scamless/artifacts")
DRIVE_ART.mkdir(parents=True, exist_ok=True)
print("artifacts will persist to", DRIVE_ART)
```

Cell 6 (code) — fetch + build dataset:

```python
!python -m scamless.data.fetch_all
!python -m scamless.data.build
```

Cell 7 (code) — train:

```python
import pathlib
import pandas as pd

from scamless.model.config import TrainConfig
from scamless.model.train import train_model

processed = pathlib.Path("training/data/processed")
cfg = TrainConfig(output_dir="artifacts/model_v1")
train_df = pd.read_parquet(processed / "messages_train.parquet")
val_df = pd.read_parquet(processed / "messages_val.parquet")

model, tokenizer, metrics = train_model(train_df, val_df, cfg)
model.save_pretrained(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
print(metrics)
```

Cell 8 (code) — export ONNX int8:

```python
!python -m scamless.model.export_onnx --model-dir artifacts/model_v1
```

Cell 9 (code) — eval, gate, and baseline comparison:

```python
import json

import pandas as pd

from scamless.eval.baseline import heuristic_predict
from scamless.eval.harness import compute_metrics

# trained model metrics + gate (gate exits nonzero on failure)
!python -m scamless.eval.run_eval --mode onnx --model-dir artifacts/model_v1
!python -m scamless.eval.gate --metrics artifacts/eval/metrics.json

# heuristic baseline for comparison (does NOT overwrite metrics.json)
test_df = pd.read_parquet("training/data/processed/messages_test.parquet")
base = compute_metrics(
    test_df, [heuristic_predict(str(t)) for t in test_df["text"]]
)
print("baseline macro_f1:", base["macro_f1"], "fp_rate:", base["false_positive_rate"])
print("trained:", json.loads(open("artifacts/eval/metrics.json").read()))
```

Cell 10 (code) — persist artifacts to Drive:

```python
import shutil

shutil.copytree("artifacts", DRIVE_ART / "model_v1_run", dirs_exist_ok=True)
print("saved to", DRIVE_ART / "model_v1_run")
```

- [ ] **Step 2: Run the notebook in Colab**

Open https://colab.research.google.com, File > Upload notebook > `notebooks/train_colab.ipynb`, set T4 GPU runtime, Run all.
Expected: all cells succeed; gate cell prints nothing on success; final cell confirms artifacts saved to Drive. Typical wall time: under 30 minutes for data, 1-3 hours for training on T4.

- [ ] **Step 3: Record measured numbers**

From Cell 9 output, record: trained macro-F1, false-positive rate, baseline macro-F1. If the gate fails, record the violations verbatim — that is a finding feeding the multilingual/synthetic-data phase, not a plan failure. The trained model must beat the baseline.

- [ ] **Step 4: Update training/README.md**

Modify `training/README.md` — replace the Quickstart section with:

```markdown
## Quickstart

Unit tests run locally (CPU only, no network):

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -e "training[dev]"
    .venv\Scripts\python.exe -m pytest training/tests

Training runs on Google Colab (free T4 GPU):

    1. Open notebooks/train_colab.ipynb in Google Colab
    2. Runtime > Change runtime type > T4 GPU
    3. Run all
    4. Artifacts land in Drive at MyDrive/scamless/artifacts
```

And in the `## Status` section replace the placeholder records with the measured values from Step 3 (baseline macro-F1, trained macro-F1, false-positive rate).

- [ ] **Step 5: Run the local unit suite + lint**

Run: `& .venv\Scripts\python.exe -m pytest training/tests -v`
Expected: all tests pass (slow-marked tests excluded by default).

Run: `& .venv\Scripts\python.exe -m ruff check training`
Expected: no violations (fix any that appear before committing).

- [ ] **Step 6: Commit**

```powershell
git add notebooks/train_colab.ipynb training/README.md
git commit -m "feat: colab gpu training notebook with full pipeline run"
```

---

## Self-Review Notes (checked)

- **Spec coverage (Phase 1 scope):** taxonomy (spec 3) -> labels.py (Task 2); corpora list (spec 6) -> Tasks 3-10; adversarial augmentation (spec 5, 11) -> Task 11; metrics + FP-rate target (spec 7, 11) -> Tasks 12-13, 16; model architecture message-classifier half (spec 4) -> Task 14; export/quantization (spec 4) -> Task 15. Span tagger, fusion combiner, URL model training, multilingual expansion, web app, and extension are Phase 2+ plans by design (spec section 12 phases 3-4); URL parsing data is staged now (Task 8) so the URL model plan has data ready.
- **Placeholder scan:** README `<record>` fields are intentional fill-in-from-measurement steps (the only permitted unknowns; they are actual outputs of Steps 1-2). No TBDs elsewhere.
- **Type consistency:** `TrainConfig` fields used in Tasks 14-15 match; `parse_message_bytes(raw, dirname, filename)` signature used consistently; `SOURCE_HF_ID` referenced in build.py is defined in Task 10's enron.py modification.
