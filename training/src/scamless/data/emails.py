"""Email parsing shared by SpamAssassin and Nazario corpora."""

import codecs
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
    try:
        codecs.lookup(charset)
    except (LookupError, ValueError):
        charset = "utf-8"
    return payload.decode(charset, errors="replace")
