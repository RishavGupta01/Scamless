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
