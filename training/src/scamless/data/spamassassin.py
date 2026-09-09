"""Parser for the SpamAssassin public corpus layout (spam_*/ easy_ham_* dirs)."""

from scamless.data.emails import extract_body
from scamless.data.schemas import make_message

SOURCE = "spamassassin"
MIN_BODY_CHARS = 40

# Keyword markers for enriching weak generic_spam labels. SpamAssassin spam is
# one undifferentiated bucket; a large share of it is stylistically phishing
# ("verify your account", "confirm your identity"). Without enrichment the
# model receives contradictory supervision: phishing-styled text labeled
# generic_spam-only teaches it to NOT fire on real phishing.
PHISHING_MARKERS = [
    "verify your account",
    "confirm your identity",
    "validate your account",
    "update your billing information",
    "unusual activity",
    "account has been suspended",
    "account will be suspended",
    "click here to login",
    "log in to your account",
    "sign in to confirm",
    "restore your account",
    "limited access",
    "password expire",
]


def classify_dir(dirname: str) -> list[str]:
    if dirname.startswith("spam"):
        return ["generic_spam"]
    return []


def enrich_labels(text: str, labels: list[str]) -> list[str]:
    if "generic_spam" not in labels:
        return labels
    lowered = text.lower()
    if any(marker in lowered for marker in PHISHING_MARKERS):
        return labels + ["phishing"]
    return labels


def parse_message_bytes(raw: bytes, dirname: str, filename: str) -> dict | None:
    text = extract_body(raw)
    if len(text) < MIN_BODY_CHARS:
        return None
    labels = enrich_labels(text, classify_dir(dirname))
    return make_message(text, labels, SOURCE, language="en")
