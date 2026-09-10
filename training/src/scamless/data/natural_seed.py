"""Loader for LLM-generated natural seed data (committed JSONL files).

This is the Phase 3 naturalness layer: natural phrasing, slang, typos, and
code-mixing that template slot-filling cannot produce. Files live in
training/assets/natural_seed/*.jsonl, one JSON object per line:

  {"text": "...", "labels": ["otp_request"], "lang": "hi"}
  {"text": "...", "labels": [], "lang": "hinglish"}   # hard negative

Labels are validated against the schema at load time; rows with unknown
labels are rejected loudly (a typo must not silently corrupt training).
"""

import json
import pathlib

from scamless.data.schemas import make_message
from scamless.labels import label_index

SEED_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "natural_seed"
SOURCE_PREFIX = "natural_llm"
MIN_TEXT_CHARS = 20


def load_seed_files(seed_dir: pathlib.Path | None = None) -> list[dict]:
    directory = seed_dir or SEED_DIR
    records: list[dict] = []
    files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    for path in files:
        rejected = 0
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = str(obj["text"]).strip()
                labels = list(obj["labels"])
                lang = str(obj["lang"]).strip()
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path.name}:{line_no}: malformed seed row ({exc})") from exc
            for label in labels:
                try:
                    label_index(label)  # unknown label must fail loudly, never silently train
                except KeyError as exc:
                    raise ValueError(
                        f"{path.name}:{line_no}: unknown label '{label}' in seed row"
                    ) from exc
            if len(text) < MIN_TEXT_CHARS:
                rejected += 1
                continue
            records.append(
                make_message(text, labels, f"{SOURCE_PREFIX}_{lang}", language=lang)
            )
        if rejected:
            print(f"{path.name}: skipped {rejected} too-short rows")
    if not files:
        print("no natural_seed files found (expected in training/assets/natural_seed)")
    return records


def collect() -> list[dict]:
    return load_seed_files()
