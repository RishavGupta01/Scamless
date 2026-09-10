"""Text normalization applied to every record before dedupe/splits.

Goals:
1. Strip HTML remnants (some corpora keep HTML-only bodies).
2. Truncate quoted reply/forward chains - they leak thread context across
   rows and dilute the actual message signal (the seven-datasets corpus card
   warns about exactly this).
3. Collapse whitespace.

Conservative by design: if cleaning shrinks a text below MIN_CHARS, the
original text is kept rather than producing a junk record.
"""

import html
import re

MIN_CHARS = 40

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_ORIG_RE = re.compile(r"-{3,}\s*original message\s*-{3,}", re.IGNORECASE)
_WROTE_RE = re.compile(r"\n-? ?on .{0,120}wrote:", re.IGNORECASE)
_REPLY_SEP_RE = re.compile(r"\n-{3,}\s*(reply|forwarded|forward message)", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = _TAG_RE.sub(" ", text)

    cut = None
    m = _ORIG_RE.search(text)
    if m:
        cut = m.start()
    m2 = _WROTE_RE.search(text)
    if m2 and (cut is None or m2.start() < cut):
        cut = m2.start()
    m3 = _REPLY_SEP_RE.search(text)
    if m3 and (cut is None or m3.start() < cut):
        cut = m3.start()
    if cut is not None and cut >= MIN_CHARS:
        text = text[:cut]

    text = _WS_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    text = text.strip()
    if len(text) < MIN_CHARS:
        return ""
    return text
