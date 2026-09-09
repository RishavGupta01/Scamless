"""Span annotations -> token-aligned tag masks for the tactic-span head.

JSONL input format (one per line):
  {"text": "...", "spans": [{"start": 0, "end": 12, "tag": "urgency"}, ...]}

Character offsets are converted to token positions using the fast tokenizer's
offset mapping. Special tokens and padding become -100 (ignored by loss).
This makes Phase 4 purely a data problem: drop a spans JSONL in, run the
pipeline, done.
"""

import json
import pathlib

import numpy as np
from transformers import PreTrainedTokenizerBase

from scamless.labels import TACTIC_TAGS

IGNORE = -100
_TAG_INDEX = {tag: i for i, tag in enumerate(TACTIC_TAGS)}


def load_spans(path: str) -> list[dict]:
    records = []
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spans file not found: {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "text" not in rec or "spans" not in rec:
            continue
        records.append(rec)
    return records


def align_spans(
    text: str,
    spans: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
) -> np.ndarray:
    """Return (max_len,) int array; -100 = ignore, else TACTIC_TAGS index."""
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "span alignment requires a fast tokenizer (offset mapping); "
            "the trainer loads fast tokenizers by default"
        )
    mask = np.full(max_len, IGNORE, dtype=np.int64)
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end:  # special tokens / padding
            continue
        for span in spans:
            tag = span.get("tag")
            if tag not in _TAG_INDEX:
                continue
            s, e = span["start"], span["end"]
            if start < e and end > s:  # any overlap
                mask[tok_idx] = _TAG_INDEX[tag]
    return mask


def build_span_masks(
    texts: list[str],
    span_records: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
) -> list[np.ndarray]:
    """One mask per text; texts without annotations get all-IGNORE rows."""
    by_text = {rec["text"]: rec["spans"] for rec in span_records}
    return [
        align_spans(text, by_text.get(text, []), tokenizer, max_len) for text in texts
    ]
