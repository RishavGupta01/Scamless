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
